# Code taken from https://github.com/manicman1999/precision-recall-keras

BATCH_SIZE = 16
IM_SIZE = (256, 256, 3)


from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
import argparse

p = argparse.ArgumentParser(description='Evaluate Image Translation Task')
p.add_argument('--fake_dir', help='Fake directory')
p.add_argument('--real_dir', help='Real Directory')
p.add_argument('--number_of_images', help='Number of images')
args = p.parse_args()

def import_images(loc, flip = True, suffix = 'jpeg'):
    
    out = []
    cont = True
    i = 1
    print("Importing Images...")
    
    while(cont):
        try:
            temp = Image.open(loc+"/im ("+str(i)+")."+suffix+"").convert('RGB')
            temp1 = np.array(temp.convert('RGB'), dtype='float32') / 255
            temp1 = np.resize(temp1, (256, 256, 3))
            # print(temp1.shape)
            out.append(temp1)
            if flip:
                out.append(np.flip(out[-1], 1))
            
            i = i + 1
        except:
            cont = False
        
    print(str(i-1) + " images imported.")
            
    return np.array(out)

def get_rand(array, amount):
    
    idx = np.random.randint(0, array.shape[0], amount)
    return array[idx]

def print_image(image):
    plt.imshow(image)
    plt.show()

"""
N-Dimensional Manifolds
"""

def get_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def get_radius(v, points):
    
    d = np.linalg.norm(np.array([v] * points.shape[0]) - points, axis = 1)
    dist = sorted(d)
    return dist

def get_manifold(points, k = 3):
    
    radii = np.zeros([points.shape[0]])
    
    # print(points.shape[0])
    
    for i in range(points.shape[0]):
        # print(get_radius(points[i], points))
        radii[i] = get_radius(points[i], points)[k]
        
    return {"points": points,
            "radii": radii,
            "len": points.shape[0]}

def in_manifold(point, manifold):
    
    for i in range(manifold["len"]):
        if get_distance(point, manifold["points"][i]) <= manifold["radii"][i]:
            return 1
        
    return 0


def get_realism(point, manifold):
    
    maxt = 0
    
    for i in range(manifold["len"]):
        t = manifold["radii"][i] / get_distance(point, manifold["points"][i])
        if t > maxt:
            maxt = t
    
    return maxt

"""
MODEL FUNCTIONS
"""

from keras.applications.inception_v3 import InceptionV3

model = InceptionV3(include_top = False, input_shape = IM_SIZE, pooling = 'avg')

def get_activations(images):
    return model.predict(images, batch_size = BATCH_SIZE)

"""
OBTAIN VALUES
"""

def get_prec_and_recall(data_loc1, data_loc2, n = 2048, get_scores = False):
    # start = time.clock()
    images = import_images(data_loc1, False, suffix = "jpeg")
    # images = get_rand(images, n)
    
    print("Getting Activations For Real Images...")
    activations1 = get_activations(images)
    print("Calculating Manifold...")
    manifold1 = get_manifold(activations1)
    
    del images
    
    images = import_images(data_loc2, False, suffix = "jpeg")
    # images = get_rand(images, n)
    
    print("Getting Activations For Fake Images...")
    activations2 = get_activations(images)
    
    if get_scores:
        scores = np.zeros([n])
        for i in range(n):
            scores[i] = get_realism(activations2[i], manifold1)
        
        del images
        return scores
    
    print("Calculating Manifold...")
    manifold2 = get_manifold(activations2)
    
    del images
    
    precision = 0
    for i in range(n):
        precision = precision + in_manifold(activations2[i], manifold1)
    precision = precision / n
    
    recall = 0
    for i in range(n):
        recall = recall + in_manifold(activations1[i], manifold2)
    recall = recall / n
    
        
    return precision, recall

precision, recall = get_prec_and_recall(args.real_dir, args.fake_dir, int(args.number_of_images))

print("Precision is: " + str(precision))
print("Recall is: " + str(recall))