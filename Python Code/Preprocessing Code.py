#Written by Matthew Brown

#Connecting Program to Google Drive
#=====================================================
from google.colab import drive
drive.mount('/gdrive')
#=====================================================

#Imports
#===============================================================================
import cv2
import keras
import matplotlib
import pickle
import os
import random
import sklearn
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import activations
from keras import backend as K
from PIL import Image
from sklearn import metrics
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Layer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#===============================================================================

#Compiling Data to Test With
#===============================================================================
Data_Location = '/gdrive/My Drive/ERAU_REU_SUMMER_2021/ERAU_retinopathy_data_rotated'
Classifications = ["Class 1", "Class 4"]

training_data = []

def initialize_training_data():
  for classification in Classifications:
    path = os.path.join(Data_Location, classification)
    classification_num = Classifications.index(classification) # 0 = Class 1, 1 = Class 4
    for img in os.listdir(path):
      img_array = cv2.imread(os.path.join(path, img))
      training_data.append([img_array, classification_num])
initialize_training_data()
random.shuffle(training_data) #Makes sure there is no order to the data set

Image_Array = []
Label_Array = []

for features, label in training_data:
  Image_Array.append(features)
  Label_Array.append(label)

Image_Array = np.array(Image_Array).reshape(-1,600,600,3) #Note the last value is 3 as it is in RGB
#===============================================================================



#Max Pool Pre-Processing
#===============================================================================
pickle_in = open("Image_Array_0.pickle", "rb")
Image_Array = pickle.load(pickle_in)
pickle_in = open("Label_Array_0.pickle", "rb")
Label_Array = pickle.load(pickle_in)
#Scaling the Variables
#===================================
Image_Array = np.array(Image_Array / 255.0)
Label_Array = np.array(Label_Array)
#===================================

#Indexing
#==========================================
image_number = 0
#==========================================

#Setting Parameters
#==========================================
filter_x = 2
filter_y = 2
filter_z = 3                                                                    #filter size 2x2x3
input_shape = (Image_Array.shape)
nx = input_shape[1]
ny = input_shape[2]
nz = input_shape[3]

output_dim = (int(input_shape[1] / filter_x), int(input_shape[2] / filter_y), filter_z)
print(output_dim)
post_vs_image_array = np.zeros((Image_Array.shape[0], ) + output_dim)                 
#==========================================
while image_number < Image_Array.shape[0]:
  print(image_number)  
  input_image = Image_Array[image_number]                                       #Successful
  post_vs_image = np.zeros(output_dim)                                          #Successful sets to (332,240,3)      

  oldrow = 0
  oldcol = 0

  new_col_idx = 0
  new_row_idx = 0

  while (oldrow < nx - 1 and new_row_idx < post_vs_image.shape[0]):

    while (oldcol < ny - 1 and new_col_idx < post_vs_image.shape[1]):

      temp_r = tf.keras.backend.get_value(input_image[oldrow:oldrow + filter_x,oldcol:oldcol + filter_y,0])
      temp_g = tf.keras.backend.get_value(input_image[oldrow:oldrow + filter_x,oldcol:oldcol + filter_y,1])
      temp_b = tf.keras.backend.get_value(input_image[oldrow:oldrow + filter_x,oldcol:oldcol + filter_y,2])

      max_r = np.max(temp_r)
      max_g = np.max(temp_g)
      max_b = np.max(temp_b)

      post_vs_image[new_row_idx, new_col_idx, 0] = max_r
      post_vs_image[new_row_idx, new_col_idx, 1] = max_g
      post_vs_image[new_row_idx, new_col_idx, 2] = max_b

      #=========================================================================
      oldcol += filter_y
      new_col_idx += 1
    oldcol = 0
    new_col_idx = 0
    oldrow += filter_x
    new_row_idx += 1
    #===========================================================================
  post_vs_image_array[image_number] = post_vs_image
  image_number += 1


#Avg Pool Pre-Processing
#===============================================================================
pickle_in = open("Image_Array_0.pickle", "rb")
Image_Array = pickle.load(pickle_in)
pickle_in = open("Label_Array_0.pickle", "rb")
Label_Array = pickle.load(pickle_in)
#Scaling the Variables
#===================================
Image_Array = np.array(Image_Array / 255.0)
Label_Array = np.array(Label_Array)
#===================================

#Indexing
#==========================================
image_number = 0
#==========================================

#Setting Parameters
#==========================================
filter_x = 2
filter_y = 2
filter_z = 3                                                                    #filter size 2x2x3
input_shape = (Image_Array.shape)
nx = input_shape[1]
ny = input_shape[2]
nz = input_shape[3] 

output_dim = (int(input_shape[1] / filter_x), int(input_shape[2] / filter_y), filter_z)
print(output_dim)
post_vs_image_array = np.zeros((Image_Array.shape[0], ) + output_dim)                 
#==========================================
while image_number < Image_Array.shape[0]:
  print(image_number)  
  input_image = Image_Array[image_number]                                       #Successful
  post_vs_image = np.zeros(output_dim)                                          #Successful sets to (332,240,3)      

  oldrow = 0
  oldcol = 0

  new_col_idx = 0
  new_row_idx = 0

  while (oldrow < nx - 1 and new_row_idx < post_vs_image.shape[0]):

    while (oldcol < ny - 1 and new_col_idx < post_vs_image.shape[1]):

      temp_r = tf.keras.backend.get_value(input_image[oldrow:oldrow + filter_x,oldcol:oldcol + filter_y,0])
      temp_g = tf.keras.backend.get_value(input_image[oldrow:oldrow + filter_x,oldcol:oldcol + filter_y,1])
      temp_b = tf.keras.backend.get_value(input_image[oldrow:oldrow + filter_x,oldcol:oldcol + filter_y,2])

      avg_r = np.average(temp_r)
      avg_g = np.average(temp_g)
      avg_b = np.average(temp_b)

      post_vs_image[new_row_idx, new_col_idx, 0] = avg_r
      post_vs_image[new_row_idx, new_col_idx, 1] = avg_g
      post_vs_image[new_row_idx, new_col_idx, 2] = avg_b

      #=========================================================================
      oldcol += filter_y
      new_col_idx += 1
    oldcol = 0
    new_col_idx = 0
    oldrow += filter_x
    new_row_idx += 1
    #===========================================================================
  post_vs_image_array[image_number] = post_vs_image
  image_number += 1


#Caluculating Variable Stride Out Shape
#===============================================================================
def CalcVSOutShape(input_shape,              #(q,x,y,z)
                   step_v = [4, 1, 4],       #Vertical Step
                   step_h = [4, 1, 4],          #Horizontal Step
                   frac_v = [1/3, 1/3, 1/3], 
                   frac_h = [1/3, 1/3, 1/3]):
  #print('Input Shape:', input_shape)
  (q, nx, ny, nz) = input_shape                 #Sets the nx, ny, nz = input x,y,z
  #print(nx)                                 #600
  #print(ny)                                 #600
  #print(nz)                                 #3
  cum_frac_v = np.cumsum(frac_v)             #Return the cumulative sum of the elements along a given axis.
  cum_frac_h = np.cumsum(frac_h)
  #print(cum_fraction_v)                     #[0.33333333 0.66666667 1.]
  #print(cum_fraction_h)                     #[0.25 1.]
  frac_bounds_v = cum_frac_v * (nx-0) - 1
  frac_bounds_h = cum_frac_h * (ny-0) - 1
  #print(frac_bounds_v)                      #[199. 399. 599.]
  #print(frac_bounds_h)                      #[149. 599.]

  out_h = 0
  out_v = 0
  idx_low = 0

  for frac_idx in range(len(frac_h)):
    idx_span = min(frac_bounds_h[frac_idx], ny - 3) - idx_low                   
    divisor_h, remander_h = divmod(idx_span, step_h[frac_idx])
    out_h += int(divisor_h) + 1
    idx_low += (int(divisor_h) + 1) * step_h[frac_idx]

  idx_low = 0

  for frac_idx in range(len(frac_v)):
    idx_span = min(frac_bounds_v[frac_idx], nx - 3) - idx_low  
    divisor_v, remainder_v = divmod(idx_span, step_v[frac_idx])
    out_v += int(divisor_v) + 1
    idx_low += (int(divisor_v) + 1) * step_v[frac_idx]

  #print(out_v)             #332
  #print(out_h)             #240 
  #print(nz)                #3
  return (out_v, out_h, nz)
#===============================================================================

#Variable Stride
#===============================================================================
pickle_in = open("Image_Array_0.pickle", "rb")
Image_Array = pickle.load(pickle_in)
pickle_in = open("Label_Array_0.pickle", "rb")
Label_Array = pickle.load(pickle_in)
#Scaling the Variables
#===================================
Image_Array = np.array(Image_Array / 255.0)
Label_Array = np.array(Label_Array)
#===================================

#Indexing
#==========================================
image_number = 0
#==========================================

#Setting Parameters
#==========================================
k1 = np.array([[1, 2, 1],
               [2, 4, 2],
               [1, 2, 1]])
step_v = [2, 3, 1, 3, 2]      #Vertical Step
step_h = [3,2]       #Horizontal Step
frac_v = [1/5, 1/5, 1/5, 1/5, 1/5]
frac_h = [1/5, 4/5]

input_shape = (Image_Array.shape)
nx = input_shape[1]
ny = input_shape[2]
nz = input_shape[3] 

output_dim = CalcVSOutShape(input_shape, step_h = step_h, step_v = step_v, frac_v = frac_v, frac_h = frac_h)
print(output_dim)
kernel = np.repeat(k1[:,:,np.newaxis], nz, axis = 2)
post_vs_image_array = np.zeros((Image_Array.shape[0], ) + output_dim)                 
#==========================================
while image_number < Image_Array.shape[0]:
  print(image_number)
  input_image = Image_Array[image_number]                                       #Successful
  post_vs_image = np.zeros(output_dim)                                          #Successful sets to (332,240,3)      

  mystep_v = 0
  mystep_h = 0
  myfrac_v = 0
  myfrac_h = 0

  oldrow = 0
  oldcol = 0

  new_col_idx = 0
  new_row_idx = 0

  while (oldrow < nx - 2 and new_row_idx < post_vs_image.shape[0]):  
    while (oldcol < ny - 2 and new_col_idx < post_vs_image.shape[1]):                    
      temp = tf.keras.backend.get_value(input_image[oldrow:oldrow + 3, oldcol:oldcol + 3, :])
      temp = (temp * kernel)                                                           
      post_vs_image[new_row_idx, new_col_idx, :] = np.sum(np.sum(temp,axis = 0), axis = 0)
      #=========================================================================
      if oldcol > (np.sum(frac_h[0:myfrac_h + 1]) * nx - 1):
        mystep_h += 1
        myfrac_h += 1
      oldcol += step_h[mystep_h]
      new_col_idx += 1
    if oldrow > (np.sum(frac_v[0:myfrac_v + 1]) * ny - 1):
      mystep_v += 1
      myfrac_v += 1
    oldcol = 0
    mystep_h = 0
    myfrac_h = 0
    oldrow += step_v[mystep_v]
    new_col_idx = 0
    new_row_idx += 1
    #===========================================================================
  post_vs_image_array[image_number] = post_vs_image
  image_number += 1


pickle_out = open("Image_Array.pickle", "wb")
pickle.dump(Image_Array, pickle_out)
pickle_out.close()

pickle_out = open("label_array.pickle", "wb")
pickle.dump(Label_Array, pickle_out)
pickle_out.close()





