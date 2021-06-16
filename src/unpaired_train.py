#snippets of this code were taken from tensorflow.org

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow_datasets as tfds
import os
import time
import datetime
from matplotlib import pyplot as plt
from IPython import display
from tensorflow_addons.layers import InstanceNormalization

from src.generator import Generator
from src.discriminator import Discriminator
from src.unpaired_prepoc import preprocess_image_train, preprocess_image_test
from src.utils import calc_cycle_loss, generate_images_unpaired, generator_loss, discriminator_loss, generator_loss_cyclegan, calc_cycle_loss, identity_loss

URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'

zip_path = keras.utils.get_file('facades.tar.gz',origin=URL, extract=True)

PATH = os.path.join(os.path.dirname(zip_path), 'facades/')

OUTPUT_CHANNELS = 3
LAMBDA = 100
BUFFER_SIZE = 1000
BATCH_SIZE = 8
EPOCHS = 150
AUTOTUNE = tf.data.AUTOTUNE

@tf.function
def train_step(real_x, real_y, generator_g, generator_f,discriminator_x, discriminator_y,summary_writer, optimizer, ckpt_manager):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.
    
    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = generator_loss_cyclegan(disc_fake_y)
    gen_f_loss = generator_loss_cyclegan(disc_fake_x)
    
    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
    
    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

    ratio = 0.5
    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x, ratio)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y, ratio)
  
  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)
  
  discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)
  
  # Apply the gradients to the optimizer
  optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

  optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))
  
  optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))
  
  optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))

def fit(train_horses, train_zebras, sample_horse, generator_g, generator_f,discriminator_x, discriminator_y,summary_writer, optimizer, ckpt_manager):
  for epoch in range(EPOCHS):
    start = time.time()

    n = 0
    for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
      train_step(image_x, image_y, generator_g, generator_f,discriminator_x, discriminator_y,summary_writer, optimizer, ckpt_manager)
      if n % 10 == 0:
        print ('.', end='')
      n += 1

    display.clear_output(wait=True)
    # Using a consistent image (sample_horse) so that the progress of the model
    # is clearly visible.
    generate_images_unpaired(generator_g, sample_horse)

    if (epoch + 1) % 5 == 0:
      ckpt_save_path = ckpt_manager.save()
      print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                          ckpt_save_path))

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))



def load_unpaired_data():
  dataset, metadata = tfds.load('cycle_gan/horse2zebra',with_info=True, as_supervised=True)
  train_horses, train_zebras = dataset['trainA'], dataset['trainB']
  test_horses, test_zebras = dataset['testA'], dataset['testB']
  train_horses = train_horses.map(preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  train_zebras = train_zebras.map(preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  test_horses = test_horses.map(preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  test_zebras = test_zebras.map(preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  
  return train_horses, train_zebras, test_horses, test_zebras


def train_unpaired():
  train_horses, train_zebras, test_horses, test_zebras = load_unpaired_data()
  sample_horse = next(iter(train_horses))
  sample_zebra = next(iter(train_zebras)) 
  generator_g = Generator(skip = True, norm_type='instancenorm')
  generator_f = Generator(skip = True, norm_type='instancenorm')
  discriminator_x = Discriminator(norm_type='instancenorm', target_flag=False)
  discriminator_y = Discriminator(norm_type='instancenorm', target_flag=False)
  
  optimizer = Adam(2e-4, beta_1=0.5)

  checkpoint_path = "./checkpoints/train"
  ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=optimizer,
                           generator_f_optimizer=optimizer,
                           discriminator_x_optimizer=optimizer,
                           discriminator_y_optimizer=optimizer)
  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

  if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)

  print ('Latest checkpoint restored!!')
  log_dir="logs/"

  summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  fit(train_horses, train_zebras, sample_horse, generator_g, generator_f,discriminator_x, discriminator_y,summary_writer, optimizer, ckpt_manager)