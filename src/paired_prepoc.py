#snippets of this code were taken from tensorflow.org

import numpy as np
import tensorflow as tf
from tensorflow import keras

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
EXPAND_WIDTH = 286
EXPAND_HEIGHT = 286
MAX_PIXEL_VALUE = 255
NORM_CONSTANT = MAX_PIXEL_VALUE / 2

def load_image(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  img_width = tf.shape(image)[1]

  img_width = img_width // 2
  domain_1_image = image[:, img_width:, :]
  domain_2_image = image[:, :img_width, :]

  domain_1_image = tf.cast(domain_1_image, tf.float32)
  domain_2_image = tf.cast(domain_2_image, tf.float32)

  return domain_1_image, domain_2_image


@tf.function()
def preprocess_training_image(domain_1_image, domain_2_image):
  # resizing to 286 x 286 x 3
  domain_1_image = tf.image.resize(domain_1_image, [EXPAND_HEIGHT, EXPAND_WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  # random crop back to 256 x 256 x 3
  domain_1_image = tf.image.random_crop(domain_1_image, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
  # normalize between [-1, 1]
  domain_1_image = (domain_1_image / NORM_CONSTANT) - 1  
  # randomly flip image horizontally
  if np.random.randint(100) > 50:
    domain_1_image = tf.image.flip_left_right(domain_1_image)

  # perform same preprocessing steps on domain 2 image
  domain_2_image = tf.image.resize(domain_2_image, [EXPAND_HEIGHT, EXPAND_WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  domain_2_image = tf.image.random_crop(domain_2_image, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
  domain_2_image = (domain_2_image / NORM_CONSTANT) - 1
  if np.random.randint(100) > 50:
    domain_2_image = tf.image.flip_left_right(domain_2_image)

  return domain_1_image, domain_2_image



@tf.function()
def preprocess_test_image(domain_1_image, domain_2_image):
  # resize domain 1 image to 256 x 256 x 3
  domain_1_image = tf.image.resize(domain_1_image, [IMAGE_HEIGHT, IMAGE_WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  # normalize image in [-1, 1]
  domain_1_image = (domain_1_image / NORM_CONSTANT) - 1 

  # perform same preprocessing steps on domain 2 image
  domain_2_image = tf.image.resize(domain_2_image, [IMAGE_HEIGHT, IMAGE_WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  domain_2_image = (domain_2_image / NORM_CONSTANT) - 1

  return domain_1_image, domain_2_image


def load_training_image(image_file):
  # load domain 1 and domain 2 image from train dataset
  domain_1_image, domain_2_image = load_image(image_file)
  domain_1_image, domain_2_image = preprocess_training_image(domain_1_image, domain_2_image)
  
  return domain_1_image, domain_2_image



def load_test_image(image_file):
  # load domain 1 and domain 2 image from test dataset
  domain_1_image, domain_2_image = load_image(image_file)
  domain_1_image, domain_2_image = preprocess_test_image(domain_1_image, domain_2_image)

  return domain_1_image, domain_2_image