import os
PROJECT_PATH='/content/drive/My Drive/unet_pipeline/Pipeline_unet-20200227T062956Z-001/Pipeline_unet'
from importlib.machinery import SourceFileLoader
dilatednet = SourceFileLoader('dilatednet', os.path.join(PROJECT_PATH, 'dilatednet.py')).load_module()
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D
from tensorflow.python.keras.optimizers import Adadelta, Nadam
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.utils import multi_gpu_model, plot_model
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.preprocessing import image
import tensorflow as tf
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.preprocessing import image
import cv2
#from multiclassunet import Unet
from dilatednet import DilatedNet
import tqdm
import time
from tensorflow.python.keras.utils import Sequence
from moviepy.editor import VideoFileClip, ImageSequenceClip



unet = DilatedNet(256, 256, 3, True, True)
unet.load_weights('/content/drive/My Drive/unet_pipeline/Pipeline_unet-20200227T062956Z-001/Pipeline_unet/pdilated_0.97901.h5')
print("Weight loaded succesfully...")


color_map = {
 '0': [0,0,255], #car
 '1': [0,0,255], #bike
 '2': [0,255,0] #bg
}



def pp(image):
    alpha = 0
    dims = image.shape
    print(dims)
    x = cv2.resize(image, (256, 256))
    x = np.float32(x)/255.
    z = unet.predict(np.expand_dims(x, axis=0))
    print(np.shape(z))
    z = np.squeeze(z)
    z = z.reshape(256, 256, 3)
    z = cv2.resize(z, (dims[1], dims[0]))
    y = np.argmax(z, axis=2)
    print(np.shape(y))
    img_color = image.copy()   
    for i in range(dims[0]):
        for j in range(dims[1]):
            img_color[i, j] = color_map[str(y[i, j])]
    cv2.addWeighted(image, alpha, img_color, 1-alpha, 0, img_color)
    return img_color



def touch(unet_result, original_img):
    original_img = original_img.astype(np.uint8)
    original_img_gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    original_img_gray_copy = original_img_gray.copy()

    result_gray=cv2.cvtColor(unet_result, cv2.COLOR_RGB2GRAY)
    
    #count = np.unique(result_gray, return_counts=True)
    #print(count)

    for i,j in enumerate(result_gray):
        for l,m in enumerate(j):
            if m > 28 and m < 30:
                original_img_gray_copy[i][l] = 0

    car=cv2.inRange(result_gray, 0, 77)
    cv2.imwrite('/content/drive/My Drive/unet_pipeline/Pipeline_unet-20200227T062956Z-001/Pipeline_unet/car_final.jpg',car)
    
    #cv2.imshow('car', car)
    #cv2.waitKey(0)

    ret, border_final = cv2.threshold(original_img_gray_copy,0, 255, cv2.THRESH_TOZERO)

    #kernel=np.ones((2,2),np.uint8)
    #border_final=cv2.morphologyEx(border_final,cv2.MORPH_OPEN,kernel)
    
    cv2.imwrite('/content/drive/My Drive/unet_pipeline/Pipeline_unet-20200227T062956Z-001/Pipeline_unet/border_final2.jpg',border_final)

    #cv2.imshow('border', opening)
    #cv2.waitKey(0)
    
    #tes=np.asarray(border)
    #print(np.shape(tes))
    #tes1=np.asarray(car_final)
    #print(np.shape(tes1))
    intersect=cv2.bitwise_and(border_final,car)
    print(np.sum(intersect))

    '''
    y=np.sum(tes)
    #s1=y
    s = add(tes,tes1)
    if s>y:
        print("Car touched the Border")
    '''
    
'''   
def add(tes,tes1):
    
    x=np.add(tes,tes1)
    print(x)
    s=0
    for i in x:
      for j in i:
        s+=j
    return s
'''
capture=cv2.VideoCapture('/content/drive/My Drive/unet_pipeline/Pipeline_unet-20200227T062956Z-001/Pipeline_unet/17-8206-2018MC-Tt.asf')
while (capture.isOpened()):
  ret,test_orig=capture.read()
  #test_orig = image.load_img('/content/drive/My Drive/unet_pipeline/Pipeline_unet-20200227T062956Z-001/Pipeline_unet/frame70.jpg')
  test = image.img_to_array(test_orig)
  unet_result = pp(test)
  final_res=touch(unet_result, test)
#i=9
