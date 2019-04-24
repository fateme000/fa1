# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#gereftan list pardazande ha
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices()) 
import keras
import tensorflow

'''
deep learning model in keras:
    1.amade kardan data(train/validation/test)
    2.ijad laye ha va model
    3.tanzim parametrha (loss & optimization...)
    4.train model(using fit())
    
'''
from keras.datasets import mnist

(train_img ,train_label), (test_img, test_label) = mnist.load_data()

#un lib miad tu train test mizare baad yeseri label 0 ta 9 dar nazar migire
print("treain_img dim:" , train_img.ndim)
print("train_img shape" , train_img.shape)
print("train_img type:" , train_img.dtype)
#namayesh tasvir
import matplotlib.pyplot as plt
digit= train_img[4]
plt.imshow(digit , cmap='binary')


my_data=train_img[10:100]#90ta dade mide
my_data = train_img[10:100,:,:]
#har balayi k sare dade miarim sare labelam bayad biarim




