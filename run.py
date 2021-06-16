import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Dropout, BatchNormalization, ReLU, LeakyReLU, Concatenate, Input, ZeroPadding2D
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import argparse

import os
import time

from matplotlib import pyplot as plt
from IPython import display
from tensorflow_addons.layers import InstanceNormalization

from src.generator import Generator
from src.paired_train import train_paired, load_paired_data, BATCH_SIZE
from src.unpaired_train import train_unpaired, load_unpaired_data

p = argparse.ArgumentParser(description='Run Image Translation Task')
p.add_argument('--train_type', help='Training type')
args = p.parse_args()

#Make data/fake and data/real directory to store the fake and real images
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('data/fake/'):
    os.makedirs('data/fake')
if not os.path.exists('data/real/'):
    os.makedirs('data/real')
path_fake = 'data/fake/'
path_real = 'data/real/'

if args.train_type == "paired":
    generator = train_paired()
    train_data, test_data = load_paired_data()
    test_length = 0
    for ex_input in test_data:
      test_length += 1
    index = 0
    for ex_input, ex_target in test_data.take(min(256/BATCH_SIZE, test_length - 1)):
      prediction = generator(ex_input)
      for i in range(BATCH_SIZE):
        tf.keras.preprocessing.image.save_img(path_fake + 'im ('+ str(index+1) + ').jpeg', prediction[i])
        tf.keras.preprocessing.image.save_img(path_real + 'im ('+ str(index+1) + ').jpeg', ex_target[i])
        index = index + 1

elif args.train_type == "unpaired":
    train_unpaired()
else:
    print('Please enter correct arguments: "paired" or "unpaired"')






