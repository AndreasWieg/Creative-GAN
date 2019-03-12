import numpy as np
import random
import os
import sys
import re
from PIL import Image
'''
________________________________________________________________________________
Utilities for the Creative-GAN
'''
#save image
def save_image(image,name,counter):
    image = np.asarray(image)
    image = Image.fromarray(image)
    image.save(name + "%d" % counter)

#resize image to a given size
def resize_image(url,size):
    training_data = []
    for file in os.listdir(url):
        image = Image.open(url+file)
        image = image.resize(size)
        a = np.asarray(image)
        training_data.append(a)
    return training_data

#shuffle the data
def shuffle_data(training_data):
     np.random.shuffle(training_data)
     return training_data

#load training data from the directory
def load_data_art():
    training_data = []
    counter = 1
    if not os.path.exists("C:/Users/Andreas/Desktop/C-GAN/new_data"):
        os.mkdir("C:/Users/Andreas/Desktop/C-GAN/new_data")
        imagepath = "C:/Users/Andreas/Desktop/C-GAN/art/"
        for file in os.listdir("C:/Users/Andreas/Desktop/C-GAN/art/"):
            if file.endswith('.jpg'):
                image = Image.open(imagepath+file)
                a = np.asarray(image)
                k = a.shape
                print(k)
                l= k[0] // 8
                w = k[1] //8
                for j in range(0,8):
                    for i in range(0,8):
                        box=(1+(i*l),1+(j*w),(i+1)*l,(j+1)*w)
                        cropped_image = image.crop(box)
                        cropped_image.save('C:/Users/Andreas/Desktop/C-GAN/new_data/%d_%s' % (i,file))
                training_data = resize_image("C:/Users/Andreas/Desktop/C-GAN/new_data/",(128,128))
    else:
        training_data = resize_image("C:/Users/Andreas/Desktop/C-GAN/new_data/",(128,128))
    return training_data
