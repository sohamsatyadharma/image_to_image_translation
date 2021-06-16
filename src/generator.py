#snippets of this code were taken from tensorflow.org

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Dropout, BatchNormalization, ReLU, LeakyReLU, Concatenate, Input, ZeroPadding2D
from tensorflow_addons.layers import InstanceNormalization

OUTPUT_CHANNELS = 3

"""
Use skip=True to use U-Net architecture, otherwise use false for encoder-decoder architecture.
Use norm_type=batchnorm or norm_type=instancenorm for batch normaliztion and instance normalization respectively.
"""
def Generator(skip=True, norm_type = 'batchnorm'):
  # Pass 256x256 image to generator
  inputs = Input(shape=[256, 256, 3])
  x = inputs
  skip_connections = []
    
  # make the encoder
  model = Sequential()
  model.add(Conv2D(64, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
  model.add(LeakyReLU())
  x = model(x)
  if skip:
    skip_connections.append(x)

  model = Sequential()
  model.add(Conv2D(128, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
  if norm_type == 'instancenorm':
    model.add(InstanceNormalization())
  else:
    model.add(BatchNormalization())
  model.add(LeakyReLU())
  x = model(x)
  if skip:
    skip_connections.append(x)

  model = Sequential()
  model.add(Conv2D(256, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
  if norm_type == 'instancenorm':
    model.add(InstanceNormalization())
  else:
    model.add(BatchNormalization())
  model.add(LeakyReLU())
  x = model(x)
  if skip:
    skip_connections.append(x)

  model = Sequential()
  model.add(Conv2D(512, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
  if norm_type == 'instancenorm':
    model.add(InstanceNormalization())
  else:
    model.add(BatchNormalization())
  model.add(LeakyReLU())
  x = model(x)
  if skip:
    skip_connections.append(x)

  model = Sequential()
  model.add(Conv2D(512, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
  if norm_type == 'instancenorm':
    model.add(InstanceNormalization())
  else:
    model.add(BatchNormalization())
  model.add(LeakyReLU())
  x = model(x)
  if skip:
    skip_connections.append(x)

  model = Sequential()
  model.add(Conv2D(512, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
  if norm_type == 'instancenorm':
    model.add(InstanceNormalization())
  else:
    model.add(BatchNormalization())
  model.add(LeakyReLU())
  x = model(x)
  if skip:
    skip_connections.append(x)

  model = Sequential()
  model.add(Conv2D(512, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
  if norm_type == 'instancenorm':
    model.add(InstanceNormalization())
  else:
    model.add(BatchNormalization())
  model.add(LeakyReLU())
  x = model(x)
  if skip:
    skip_connections.append(x)

  model = Sequential()
  model.add(Conv2D(512, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
  if norm_type == 'instancenorm':
    model.add(InstanceNormalization())
  else:
    model.add(BatchNormalization())
  model.add(LeakyReLU())
  x = model(x)
  if skip:
    skip_connections.reverse()

  # make the decoder
  model = Sequential()
  model.add(Conv2DTranspose(512, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
  if norm_type == 'instancenorm':
    model.add(InstanceNormalization())
  else:
    model.add(BatchNormalization())
  model.add(Dropout(0.5))
  model.add(ReLU())
  x = model(x)
  if skip:
    x = Concatenate()([x, skip_connections[0]])

  model = Sequential()
  model.add(Conv2DTranspose(512, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
  if norm_type == 'instancenorm':
    model.add(InstanceNormalization())
  else:
    model.add(BatchNormalization())
  model.add(Dropout(0.5))
  model.add(ReLU())
  x = model(x)
  if skip:
    x = Concatenate()([x, skip_connections[1]])

  model = Sequential()
  model.add(Conv2DTranspose(512, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
  if norm_type == 'instancenorm':
    model.add(InstanceNormalization())
  else:
    model.add(BatchNormalization())
  model.add(Dropout(0.5))
  model.add(ReLU())
  x = model(x)
  if skip:
    x = Concatenate()([x, skip_connections[2]])

  model = Sequential()
  model.add(Conv2DTranspose(512, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
  if norm_type == 'instancenorm':
    model.add(InstanceNormalization())
  else:
    model.add(BatchNormalization())
  model.add(ReLU())
  x = model(x)
  if skip:
    x = Concatenate()([x, skip_connections[3]])

  model = Sequential()
  model.add(Conv2DTranspose(256, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
  if norm_type == 'instancenorm':
    model.add(InstanceNormalization())
  else:
    model.add(BatchNormalization())
  model.add(ReLU())
  x = model(x)
  if skip:
    x = Concatenate()([x, skip_connections[4]])

  model = Sequential()
  model.add(Conv2DTranspose(128, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
  if norm_type == 'instancenorm':
    model.add(InstanceNormalization())
  else:
    model.add(BatchNormalization())
  model.add(ReLU())
  x = model(x)
  if skip:
    x = Concatenate()([x, skip_connections[5]])

  model = Sequential()
  model.add(Conv2DTranspose(64, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
  if norm_type == 'instancenorm':
    model.add(InstanceNormalization())
  else:
    model.add(BatchNormalization())
  model.add(ReLU())
  x = model(x)
  if skip:
    x = Concatenate()([x, skip_connections[6]])

  model = Sequential()
  model.add(Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), activation='tanh'))
  x = model(x)

  return tf.keras.Model(inputs=inputs, outputs=x)