#snippets of this code were taken from tensorflow.org

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Dropout, BatchNormalization, ReLU, LeakyReLU, Concatenate, Input, ZeroPadding2D
from tensorflow_addons.layers import InstanceNormalization

def Discriminator(norm_type='batchnorm', target_flag=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  input = Input(shape=[256, 256, 3], name='input_image')
  if target_flag:
    target = Input(shape=[256, 256, 3], name='target_image')
    x = Concatenate()([input, target])  # (bs, 256, 256, channels*2)
  else:
    x = input

  model = Sequential()
  model.add(Conv2D(64, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
  model.add(LeakyReLU())
  x = model(x)
  
  model = Sequential()
  model.add(Conv2D(128, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
  if norm_type == 'instancenorm':
    model.add(InstanceNormalization())
  else:
    model.add(BatchNormalization())
  model.add(LeakyReLU())
  x = model(x)

  model = Sequential()
  model.add(Conv2D(256, 4, strides=2, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02), use_bias=False))
  if norm_type == 'instancenorm':
    model.add(InstanceNormalization())
  else:
    model.add(BatchNormalization())
  model.add(LeakyReLU())
  x = model(x)

  x = ZeroPadding2D()(x)  
  x = tf.keras.layers.Conv2D(512, 4, strides=1,kernel_initializer=initializer, use_bias=False)(x)  # (bs, 31, 31, 512)
  x = BatchNormalization()(x)
  x = LeakyReLU()(x)
  x = ZeroPadding2D()(x) 
  x = Conv2D(1, 4, strides=1, kernel_initializer=tf.random_normal_initializer(0., 0.02))(x)  # (bs, 30, 30, 1)

  if target_flag:
    return tf.keras.Model(inputs=[input, target], outputs=x)
  else:
    return tf.keras.Model(inputs=input, outputs=x)
  #return tf.keras.Model(inputs=[input, target], outputs=x)