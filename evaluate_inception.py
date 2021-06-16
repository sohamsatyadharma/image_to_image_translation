# scale an array of images to a new size

###### Part of the code has been inspired and taken from the following link:
####https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/

import os, numpy as np
# example of calculating the frechet inception distance in Keras
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from src.utils import load_dir, scale_images, calculate_fid
import argparse


p = argparse.ArgumentParser(description='Evaluate Image Translation Task')
p.add_argument('--fake_dir', help='Fake directory')
p.add_argument('--real_dir', help='Real Directory')
args = p.parse_args()
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
# define two fake collections of images
images1 = load_dir(args.fake_dir)
images2 = load_dir(args.real_dir)
# print(images1.shape, images2.shape)
images1 = images1.astype('float32')
images2 = images2.astype('float32')
# resize images
images1 = scale_images(images1, (299,299,3))
images2 = scale_images(images2, (299,299,3))
print('Scaled', images1.shape, images2.shape)
# pre-process images
images1 = preprocess_input(images1)
images2 = preprocess_input(images2)
fid = calculate_fid(model, images1, images2)
print('FID (different): %.3f' % fid)