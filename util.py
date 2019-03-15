import numpy as np
import random
import os
import sys
import re
from PIL import Image
from sklearn import preprocessing
'''
________________________________________________________________________________
Utilities for the Creative-GAN
'''

def inverse_norm_image(image):
    image = (np.array(image) -1.0) * 127.5
    image = image.astype("uint8")
    return image
# normalize image between 0-1
def norm_image(image):
    return (np.array(image) / 127.5) - 1.0
#save image
def save_image(image,name,counter):
    image = np.asarray(image)
    image = np.reshape(image,(128,128,3))
    image = inverse_norm_image(image)
    image = Image.fromarray(image)
    image.save(name+"%d"% counter+".jpg","JPEG")

#resize image to a given size
def resize_image(url,size):
    training_data = []
    for file in os.listdir(url):
        image = Image.open(url+file)
        image = image.resize(size)
        image = np.asarray(image)
        image = norm_image(image)
        training_data.append(image)
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
                l= k[0] // 8
                w = k[1] //8
                for j in range(0,8):
                    for i in range(0,8):
                        box=(1+(i*l),1+(j*w),(i+1)*l,(j+1)*w)
                        cropped_image = image.crop(box)
                        cropped_image.save('C:/Users/Andreas/Desktop/C-GAN/new_data/%d_%d_%s' % (j,i,file))
                training_data = resize_image("C:/Users/Andreas/Desktop/C-GAN/new_data/",(128,128))
    else:
        training_data = resize_image("C:/Users/Andreas/Desktop/C-GAN/new_data/",(128,128))
    training_data = np.asarray(training_data)
    return training_data



def load_data_small():
    training_data = []
    counter = 1
    if not os.path.exists("C:/Users/Andreas/Desktop/C-GAN/new_data_small"):
        os.mkdir("C:/Users/Andreas/Desktop/C-GAN/new_data_small")
        imagepath = "C:/Users/Andreas/Desktop/C-GAN/skizze/"
        for file in os.listdir("C:/Users/Andreas/Desktop/C-GAN/skizze/"):
            if file.endswith('.jpg'):
                print(file)
                image = Image.open(imagepath+file)
                a = np.asarray(image)
                k = a.shape
                l= k[0] // 2
                w = k[1] //2
                for j in range(0,1):
                    for i in range(0,1):
                        box=(1+(i*l),1+(j*w),(i+1)*l,(j+1)*w)
                        cropped_image = image.crop(box)
                        cropped_image.save('C:/Users/Andreas/Desktop/C-GAN/new_data_small/%d_%d%d_%s' % (j,i,j,file))
                training_data = resize_image("C:/Users/Andreas/Desktop/C-GAN/new_data_small/",(128,128))
    #else:
        #training_data = resize_image("C:/Users/Andreas/Desktop/C-GAN/new_data/",(128,128))
    training_data = np.asarray(training_data)
    return training_data
