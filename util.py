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
# normalize image source: https://www.kaggle.com/gauss256/preprocess-images

def inverse_norm_image(image):
    image = np.clip(image, 0.0, 255.0)
    image = image.astype(np.uint8)
    return image

def norm_image(img):
    """Normalize PIL image
    Normalizes luminance to (mean,std)=(0,1), and applies a [1%, 99%] contrast stretch
    """
    img_y, img_b, img_r = img.convert('YCbCr').split()
    img_y_np = np.asarray(img_y).astype(float)
    img_y_np /= 255
    img_y_np -= img_y_np.mean()
    img_y_np /= img_y_np.std()
    scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                    np.abs(np.percentile(img_y_np, 99.0))])
    img_y_np = img_y_np / scale
    img_y_np = np.clip(img_y_np, -1.0, 1.0)
    img_y_np = (img_y_np + 1.0) / 2.0
    img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)
    img_y = Image.fromarray(img_y_np)
    img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))
    img_nrm = img_ybr.convert('RGB')
    return img_nrm


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
        image = norm_image(image)
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
                image = norm_image(image)
                a = np.asarray(image)
                k = a.shape
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
    training_data = np.asarray(training_data)
    return training_data
