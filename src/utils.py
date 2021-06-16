#snippets of this code were taken from tensorflow.org

import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from scipy.linalg import sqrtm
from keras.preprocessing.image import load_img
from numpy import trace
from numpy import iscomplexobj
from numpy import cov
from numpy import asarray
from skimage.transform import resize
from matplotlib import pyplot as plt

LAMBDA_P = 100
LAMBDA_UP = 10

binary_crossentropy = BinaryCrossentropy(from_logits=True)
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

"""
Change the value of l1_ratio and l2_ratio to perform various experiments
Currently, the values are for the L1 loss experiment
"""
def generator_loss(disc_generated_output, gen_output, target, l1_ratio = 1, l2_ratio = 0):
  gan_loss = binary_crossentropy(tf.ones_like(disc_generated_output), disc_generated_output)
  
  loss1 = tf.reduce_mean(tf.abs(target - gen_output))
  loss2 = loss2 = tf.keras.losses.MeanSquaredError()(target, gen_output)
  loss = l1_ratio * loss1 + l2_ratio * loss2
    
  generator_loss = gan_loss + (LAMBDA_P * loss)

  return generator_loss, gan_loss, loss



def discriminator_loss(disc_real_output, disc_generated_output, ratio = 1):
  real_loss = binary_crossentropy(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = binary_crossentropy(tf.zeros_like(disc_generated_output), disc_generated_output)

  discriminator_loss = real_loss + generated_loss

  return discriminator_loss * ratio



def generate_images(generator, input, target):
  predicted_image = generator(input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [input[0], target[0], predicted_image[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()


def generator_loss_cyclegan(generated):
  return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image, l1_ratio=1, l2_ratio=0):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  loss2 = tf.keras.losses.MeanSquaredError()(real_image, cycled_image)
  loss = l1_ratio * loss1 + l2_ratio * loss2
  return LAMBDA_UP * loss


def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA_UP * 0.5 * loss


def generate_images_unpaired(model, test_input):
  prediction = model(test_input)
  plt.figure(figsize=(12, 12))
  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']
  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()


def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

# prepare the inception v3 model

def load_dir(dir_name):
  files = os.listdir(dir_name)
  image_mat = np.zeros((len(files), 256, 256,3))
  for i, file in enumerate(files):
    img_path = os.path.join(dir_name,file)
    print(img_path)
    img = load_img(img_path)
    image_mat[i] = img
  return image_mat
