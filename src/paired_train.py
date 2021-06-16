#snippets of this code were taken from tensorflow.org

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from IPython import display

import os
import time
import datetime
from matplotlib import pyplot as plt
from tensorflow_addons.layers import InstanceNormalization

from src.generator import Generator
from src.discriminator import Discriminator
from src.paired_prepoc import load_training_image, load_test_image
from src.utils import generate_images, generator_loss, discriminator_loss

# load the dataset from the web
URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
zip_path = keras.utils.get_file('facades.tar.gz',origin=URL, extract=True)
PATH = os.path.join(os.path.dirname(zip_path), 'facades/')

OUTPUT_CHANNELS = 3
LAMBDA = 100
BUFFER_SIZE = 500
BATCH_SIZE = 16
EPOCHS = 150

@tf.function
def train_step(input_image, target, epoch, optimizer, generator, discriminator, summary_writer):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
  optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)


def fit(train_data, epochs, test_data, generator, discriminator, summary_writer, optimizer, checkpoint, checkpoint_prefix):
  for epoch in range(epochs):
    start = time.time()

    display.clear_output(wait=True)

    for example_input, example_target in test_data.take(1):
      generate_images(generator, example_input, example_target)
    print("Epoch: ", epoch)

    # Train
    for n, (input_image, target) in train_data.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      train_step(input_image, target, epoch, optimizer, generator, discriminator, summary_writer)
    print()

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
  checkpoint.save(file_prefix=checkpoint_prefix)



def load_paired_data():
  train_data = tf.data.Dataset.list_files(PATH+'train/*.jpg')
  train_data = train_data.map(load_training_image, num_parallel_calls=tf.data.AUTOTUNE)
  train_data = train_data.shuffle(BUFFER_SIZE)
  train_data = train_data.batch(BATCH_SIZE)


  test_data = tf.data.Dataset.list_files(PATH+'test/*.jpg')
  test_data = test_data.map(load_test_image)
  test_data = test_data.batch(BATCH_SIZE)

  return train_data, test_data


def train_paired():
  train_data, test_data = load_paired_data()
  generator = Generator()
  discriminator = Discriminator()
  optimizer = Adam(2e-4, beta_1=0.5)
  checkpoint_dir = './training_checkpoints'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator_optimizer=optimizer,
                                  discriminator_optimizer=optimizer,
                                  generator=generator,
                                  discriminator=discriminator)
  log_dir="logs/"

  summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  fit(train_data, EPOCHS, test_data, generator, discriminator, summary_writer, optimizer, checkpoint, checkpoint_prefix)
  return generator



