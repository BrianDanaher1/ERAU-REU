from google.colab import drive
drive.mount('/content/drive')

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
  print('and then re-execute this cell.')
else:
  print(gpu_info)
  
  #Imports

%tensorflow_version 2.x
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks

#Import helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle

#Train the model! (Note: takes a long time)
import time
import sklearn
import sklearn.metrics
from keras import backend as K
import datetime
from tensorflow.keras.callbacks import TensorBoard
import os

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) #Edit Name of Logs
tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

#number of epochs per run
epochnum = 150
#Number of runs per network 
num1 = 4
#Number of network structures
num2 = 3 
#Number of pooling methods 
num3 = 1

universal_fpr_keras = np.zeros((1000,6))
universal_tpr_keras = np.zeros((1000,6))
universal_auc = np.zeros((4,6))

for b in range(5):
  
    if b == 0 : 
      pickle_in = open('/content/drive/MyDrive/Embry-Riddle REU/Brian/average_pool_image.pickle', "rb")
      print('AVGPOOL')

      data_root='/content/drive/MyDrive/Embry-Riddle REU/Brian/average_pool_label_array.pickle'

      x = pickle.load(pickle_in)
      pickle_in = open('/content/drive/MyDrive/Embry-Riddle REU/Brian/label_array.pickle', "rb")
      y = pickle.load(pickle_in)

      valid_seperation = 0.2

      test_data_1 = int(200 - (200 * valid_seperation))
      test_data_2 = int(200 - (200 * valid_seperation))
      i = 0

      dim = (40,300, 300, 3)
      dim2 = (160,300, 300, 3)

      inputshape = [300,300,3]

      test_data_image = np.zeros(dim)
      test_data_label = np.zeros(40)
      train_data_image = np.zeros(dim2)
      train_data_label = np.zeros(160)

      print(test_data_2)

      for train_data_1 in range(0, test_data_2-1):
        train_data_image[i] = x[train_data_1]
        train_data_label[i] = y[train_data_1]
        i += 1

      i = 0

      for test_data_1 in range(test_data_2, 200):
        test_data_image[i] = x[test_data_1]
        test_data_label[i] = y[test_data_1]
        i += 1

      train_images = train_data_image
      train_labels = train_data_label

      test_images = test_data_image
      test_labels = test_data_label

      print(np.shape(train_images))

      glob_fpr_keras1 = []
      glob_tpr_keras1 = []
      glob_auc1 = np.zeros(num1)

      glob_fpr_keras2 = []
      glob_tpr_keras2 = []
      glob_auc2 = np.zeros(num1)

      glob_fpr_keras3 = []
      glob_tpr_keras3 = []
      glob_auc3 = np.zeros(num1)


    if b == 1 : 
      pickle_in = open('/content/drive/MyDrive/Embry-Riddle REU/Brian/max_pool_image.pickle', "rb")
      print('MAXPOOL')

      x = pickle.load(pickle_in)
      pickle_in = open('/content/drive/MyDrive/Embry-Riddle REU/Brian/label_array.pickle', "rb")
      y = pickle.load(pickle_in)

      valid_seperation = 0.2

      test_data_1 = int(200 - (200 * valid_seperation))
      test_data_2 = int(200 - (200 * valid_seperation))
      i = 0

      dim = (40,300, 300, 3)
      dim2 = (160,300, 300, 3)

      inputshape = [300,300,3]

      test_data_image = np.zeros(dim)
      test_data_label = np.zeros(40)
      train_data_image = np.zeros(dim2)
      train_data_label = np.zeros(160)

      print(test_data_2)

      for train_data_1 in range(0, test_data_2-1):
        train_data_image[i] = x[train_data_1]
        train_data_label[i] = y[train_data_1]
        i += 1

      i = 0

      for test_data_1 in range(test_data_2, 200):
        test_data_image[i] = x[test_data_1]
        test_data_label[i] = y[test_data_1]
        i += 1

      train_images = train_data_image
      train_labels = train_data_label

      test_images = test_data_image
      test_labels = test_data_label

      print(np.shape(train_images))

      glob_fpr_keras1 = []
      glob_tpr_keras1 = []
      glob_auc1 = np.zeros(num1)

      glob_fpr_keras2 = []
      glob_tpr_keras2 = []
      glob_auc2 = np.zeros(num1)

      glob_fpr_keras3 = []
      glob_tpr_keras3 = []
      glob_auc3 = np.zeros(num1)


    if b == 2 : 
      pickle_in = open('/content/drive/MyDrive/Embry-Riddle REU/Brian/variable_stride_right_image_array.pickle', "rb")
      print('VS RIGHT')

      x = pickle.load(pickle_in)
      pickle_in = open('/content/drive/MyDrive/Embry-Riddle REU/Brian/label_array.pickle', "rb")
      y = pickle.load(pickle_in)

      valid_seperation = 0.2

      test_data_1 = int(200 - (200 * valid_seperation))
      test_data_2 = int(200 - (200 * valid_seperation))
      i = 0

      dim = (40,332, 266, 3)
      dim2 = (160,332, 266, 3)

      inputshape = [332,266,3]

      test_data_image = np.zeros(dim)
      test_data_label = np.zeros(40)
      train_data_image = np.zeros(dim2)
      train_data_label = np.zeros(160)

      print(test_data_2)

      for train_data_1 in range(0, test_data_2-1):
        train_data_image[i] = x[train_data_1]
        train_data_label[i] = y[train_data_1]
        i += 1

      i = 0

      for test_data_1 in range(test_data_2, 200):
        test_data_image[i] = x[test_data_1]
        test_data_label[i] = y[test_data_1]
        i += 1

      train_images = train_data_image
      train_labels = train_data_label

      test_images = test_data_image
      test_labels = test_data_label

      print(np.shape(train_images))

      glob_fpr_keras1 = []
      glob_tpr_keras1 = []
      glob_auc1 = np.zeros(num1)

      glob_fpr_keras2 = []
      glob_tpr_keras2 = []
      glob_auc2 = np.zeros(num1)

      glob_fpr_keras3 = []
      glob_tpr_keras3 = []
      glob_auc3 = np.zeros(num1)


    if b == 3 : 
      pickle_in = open('/content/drive/MyDrive/Embry-Riddle REU/Brian/variable_stride_center_image_array.pickle', "rb")
      print('VS CENTER')

      x = pickle.load(pickle_in)
      pickle_in = open('/content/drive/MyDrive/Embry-Riddle REU/Brian/label_array.pickle', "rb")
      y = pickle.load(pickle_in)

      valid_seperation = 0.2

      test_data_1 = int(200 - (200 * valid_seperation))
      test_data_2 = int(200 - (200 * valid_seperation))
      i = 0

      dim = (40,300, 300, 3)
      dim2 = (160,300, 300, 3)

      inputshape = [300,300,3]

      test_data_image = np.zeros(dim)
      test_data_label = np.zeros(40)
      train_data_image = np.zeros(dim2)
      train_data_label = np.zeros(160)

      print(test_data_2)

      for train_data_1 in range(0, test_data_2-1):
        train_data_image[i] = x[train_data_1]
        train_data_label[i] = y[train_data_1]
        i += 1

      i = 0

      for test_data_1 in range(test_data_2, 200):
        test_data_image[i] = x[test_data_1]
        test_data_label[i] = y[test_data_1]
        i += 1

      train_images = train_data_image
      train_labels = train_data_label

      test_images = test_data_image
      test_labels = test_data_label

      print(np.shape(train_images))

      glob_fpr_keras1 = []
      glob_tpr_keras1 = []
      glob_auc1 = np.zeros(num1)

      glob_fpr_keras2 = []
      glob_tpr_keras2 = []
      glob_auc2 = np.zeros(num1)

      glob_fpr_keras3 = []
      glob_tpr_keras3 = []
      glob_auc3 = np.zeros(num1)


    if b == 4 : 
      pickle_in = open('/content/drive/MyDrive/Embry-Riddle REU/Brian/variable_stride_custom_image_array.pickle', "rb")
      print('VS CUSTOM')

      x = pickle.load(pickle_in)
      pickle_in = open('/content/drive/MyDrive/Embry-Riddle REU/Brian/label_array.pickle', "rb")
      y = pickle.load(pickle_in)

      valid_seperation = 0.2

      test_data_1 = int(200 - (200 * valid_seperation))
      test_data_2 = int(200 - (200 * valid_seperation))
      i = 0

      dim = (40,319, 279, 3)
      dim2 = (160,319, 279, 3)

      inputshape = [319,279,3]

      test_data_image = np.zeros(dim)
      test_data_label = np.zeros(40)
      train_data_image = np.zeros(dim2)
      train_data_label = np.zeros(160)

      print(test_data_2)

      for train_data_1 in range(0, test_data_2-1):
        train_data_image[i] = x[train_data_1]
        train_data_label[i] = y[train_data_1]
        i += 1

      i = 0

      for test_data_1 in range(test_data_2, 200):
        test_data_image[i] = x[test_data_1]
        test_data_label[i] = y[test_data_1]
        i += 1

      train_images = train_data_image
      train_labels = train_data_label

      test_images = test_data_image
      test_labels = test_data_label

      print(np.shape(train_images))

      glob_fpr_keras1 = []
      glob_tpr_keras1 = []
      glob_auc1 = np.zeros(num1)

      glob_fpr_keras2 = []
      glob_tpr_keras2 = []
      glob_auc2 = np.zeros(num1)

      glob_fpr_keras3 = []
      glob_tpr_keras3 = []
      glob_auc3 = np.zeros(num1)










    for a in range (num2):

      if a == 0:

        for n in range(num1):

          #THE NETWORK - Rebuilds for each run

          #This is the "convolutional base"
          model = tf.keras.Sequential()
          model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=inputshape)) #32x32x3 pixels passing 32 3x3 filters over the data
          model.add(layers.MaxPooling2D((2, 2))) #downsample
          model.add(layers.Conv2D(32, (3, 3), activation='relu')) #Passing 64 3x3 filters over the image
          model.add(layers.MaxPooling2D((2, 2))) #downsample
          model.add(layers.Conv2D(32, (3, 3), activation='relu')) #Passing 64 3x3 filters over the image
          model.add(layers.MaxPooling2D((2, 2))) #downsample
          model.add(layers.Conv2D(32, (3, 3), activation='relu')) #Passing 64 3x3 filters over the image
          model.add(layers.MaxPooling2D((2, 2))) #downsample
          model.add(layers.Conv2D(32, (3, 3), activation='relu')) #Passing 64 3x3 filters over the image
          model.add(layers.MaxPooling2D((2, 2))) #downsample

          model.add(layers.Flatten())
          model.add(layers.Dense(32, activation='relu'))
          model.add(layers.Dense(32, activation='relu'))
          model.add(layers.Dense(32, activation='relu'))
          model.add(layers.Dense(32, activation='relu'))
          model.add(layers.Dense(1, activation='sigmoid'))
          model.summary()

          model.compile(optimizer="SGD",
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'],run_eagerly=True)

          os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

          history = model.fit(train_images, train_labels, epochs=epochnum,
                            validation_data=(test_images, test_labels), batch_size=1, callbacks=[tensorboard_callback])

          test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size = 1)
          print(test_acc)

          y_pred_keras = model.predict(test_images, batch_size = 1)

          fpr_keras1, tpr_keras1, thresholds_keras = sklearn.metrics.roc_curve(test_labels,y_pred_keras)
          auc = sklearn.metrics.roc_auc_score(test_labels,y_pred_keras)

          glob_fpr_keras1.extend(fpr_keras1)
          glob_tpr_keras1.extend(tpr_keras1)
          glob_auc1[n] = auc

          tf.keras.backend.clear_session()
        #Load Tensor Board
      #===============================================================================
      #%reload_ext tensorboard
        %load_ext tensorboard
        %tensorboard --logdir logs
      #===============================================================================

        print(np.shape(universal_fpr_keras))
        print(np.shape(glob_fpr_keras1))
        print(np.shape(glob_fpr_keras2))
        print(np.shape(glob_fpr_keras3))

        for n in range(np.max(np.size(y_pred_keras))):
          if y_pred_keras[n,0] < 0.5:
            y_pred_keras[n,0] = 0

          if y_pred_keras[n,0] >= 0.5:
            y_pred_keras[n,0] = 1 

      print(tf.math.confusion_matrix(test_labels, y_pred_keras))
      print('Classification Report')

      stringy = sklearn.metrics.classification_report(test_labels,y_pred_keras,target_names = ["Class 1", "Class 4"])
      split_stringy = stringy.split()

      stringy = sklearn.metrics.classification_report(test_labels,y_pred_keras,target_names = ["Class 1", "Class 4"])
      print(stringy)

      if a == 1 :
        for n in range(num1):

          #THE NETWORK - Rebuilds for each run

          #This is the "convolutional base"
          model = tf.keras.Sequential()
          model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=inputshape)) #32x32x3 pixels passing 32 3x3 filters over the data
          model.add(layers.MaxPooling2D((2, 2))) #downsample
          model.add(layers.Dropout(.05))
          model.add(layers.Conv2D(16, (3, 3), activation='relu')) #Passing 64 3x3 filters over the image
          model.add(layers.MaxPooling2D((2, 2))) #downsample
          model.add(layers.Dropout(.05))
          model.add(layers.Conv2D(16, (3, 3), activation='relu')) #Passing 64 3x3 filters over the image
          model.add(layers.MaxPooling2D((2, 2))) #downsample
          model.add(layers.Dropout(.05))
          model.add(layers.Conv2D(16, (3, 3), activation='relu')) #32x32x3 pixels passing 32 3x3 filters over the data
          model.add(layers.MaxPooling2D((2, 2))) #downsample
          model.add(layers.Dropout(.05))
          model.add(layers.Conv2D(16, (3, 3), activation='relu')) #32x32x3 pixels passing 32 3x3 filters over the data
          model.add(layers.MaxPooling2D((2, 2))) #downsample

          model.add(layers.Flatten())
          model.add(layers.Dense(16, activation='relu'))
          model.add(layers.Dense(1, activation='sigmoid'))
          model.summary()

          model.compile(optimizer="adam",
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'],run_eagerly=True)

          os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

          history = model.fit(train_images, train_labels, epochs=epochnum,
                            validation_data=(test_images, test_labels), batch_size=1, callbacks=[tensorboard_callback])

          test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size = 1)
          print(test_acc)

          y_pred_keras = model.predict(test_images, batch_size = 1)

          fpr_keras2, tpr_keras2, thresholds_keras = sklearn.metrics.roc_curve(test_labels,y_pred_keras)
          auc = sklearn.metrics.roc_auc_score(test_labels,y_pred_keras)

          glob_fpr_keras2.extend(fpr_keras2)
          glob_tpr_keras2.extend(tpr_keras2)
          glob_auc2[n] = auc
      

          tf.keras.backend.clear_session()
        #Load Tensor Board
      #===============================================================================
      #%reload_ext tensorboard
        %load_ext tensorboard
        %tensorboard --logdir logs

        print(np.shape(universal_fpr_keras))
        print(np.shape(glob_fpr_keras1))
        print(np.shape(glob_fpr_keras2))
        print(np.shape(glob_fpr_keras3))

        for n in range(np.max(np.size(y_pred_keras))):
          if y_pred_keras[n,0] < 0.5:
            y_pred_keras[n,0] = 0

          if y_pred_keras[n,0] >= 0.5:
            y_pred_keras[n,0] = 1 

        print(tf.math.confusion_matrix(test_labels, y_pred_keras))
        print('Classification Report')

        stringy = sklearn.metrics.classification_report(test_labels,y_pred_keras,target_names = ["Class 1", "Class 4"])
        print(stringy)

      if a == 2 :
        for n in range(num1):

          #THE NETWORK - Rebuilds for each run

          model = tf.keras.Sequential()
          model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=inputshape)) #32x32x3 pixels passing 32 3x3 filters over the data
          model.add(layers.MaxPooling2D((2, 2))) #downsample
          model.add(layers.Dropout(.05))
          model.add(layers.Conv2D(64, (3, 3), activation='relu')) #Passing 64 3x3 filters over the image  
          model.add(layers.MaxPooling2D((2, 2))) #downsample
          model.add(layers.Dropout(.05))
          model.add(layers.Conv2D(64, (3, 3), activation='relu')) #Passing 64 3x3 filters over the image  
          model.add(layers.MaxPooling2D((2, 2))) #downsample
          model.add(layers.Dropout(.05))
          model.add(layers.Conv2D(64, (3, 3), activation='relu')) #32x32x3 pixels passing 32 3x3 filters over the data   
          model.add(layers.MaxPooling2D((2, 2))) #downsample
          model.add(layers.Dropout(.05))
          model.add(layers.Conv2D(64, (3, 3), activation='relu')) #32x32x3 pixels passing 32 3x3 filters over the data
          model.add(layers.MaxPooling2D((2, 2))) #downsample
          model.add(layers.Dropout(.05))
          model.add(layers.Flatten())

          model.add(layers.Dense(64, activation='relu'))
          model.add(layers.Dense(64, activation='relu'))
          model.add(layers.Dense(64, activation='relu'))
          model.add(layers.Dense(64, activation='relu'))
          model.add(layers.Dense(1, activation='sigmoid'))
          model.summary()

          model.compile(optimizer="adam",
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'],run_eagerly=True)

          os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

          history = model.fit(train_images, train_labels, epochs=epochnum,
                            validation_data=(test_images, test_labels), batch_size=1, callbacks=[tensorboard_callback])

          test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size = 1)
          print(test_acc)

          y_pred_keras = model.predict(test_images, batch_size = 1)

          fpr_keras3, tpr_keras3, thresholds_keras = sklearn.metrics.roc_curve(test_labels,y_pred_keras)
          auc = sklearn.metrics.roc_auc_score(test_labels,y_pred_keras)

          glob_fpr_keras3.extend(fpr_keras3)
          glob_tpr_keras3.extend(tpr_keras3)
          glob_auc3[n] = auc
      

          tf.keras.backend.clear_session()
        #Load Tensor Board
      #===============================================================================
      #%reload_ext tensorboard
        %load_ext tensorboard
        %tensorboard --logdir logs

        print(np.shape(universal_fpr_keras))
        print(np.shape(glob_fpr_keras1))
        print(np.shape(glob_fpr_keras2))
        print(np.shape(glob_fpr_keras3))

        for n in range(np.max(np.size(y_pred_keras))):
          if y_pred_keras[n,0] < 0.5:
            y_pred_keras[n,0] = 0

          if y_pred_keras[n,0] >= 0.5:
            y_pred_keras[n,0] = 1 

        print(tf.math.confusion_matrix(test_labels, y_pred_keras))
        print('Classification Report')

        stringy = sklearn.metrics.classification_report(test_labels,y_pred_keras,target_names = ["Class 1", "Class 4"])
        print(stringy)


        if b == 0:

          universal_fpr_keras[0:np.size(glob_fpr_keras1), 0] = glob_fpr_keras1
          universal_fpr_keras[0:np.size(glob_fpr_keras2), 1] = glob_fpr_keras2
          universal_fpr_keras[0:np.size(glob_fpr_keras3), 2] = glob_fpr_keras3

          universal_tpr_keras[0:np.size(glob_tpr_keras1), 0] = glob_tpr_keras1
          universal_tpr_keras[0:np.size(glob_tpr_keras2), 1] = glob_tpr_keras2
          universal_tpr_keras[0:np.size(glob_tpr_keras3), 2] = glob_tpr_keras3


          universal_auc[:, 0] = glob_auc1
          universal_auc[:, 1] = glob_auc2
          universal_auc[:, 2] = glob_auc3

        if b == 1:

          universal_fpr_keras[0:np.size(glob_fpr_keras1), 3] = glob_fpr_keras1
          universal_fpr_keras[0:np.size(glob_fpr_keras2), 4] = glob_fpr_keras2
          universal_fpr_keras[0:np.size(glob_fpr_keras3), 5] = glob_fpr_keras3

          universal_tpr_keras[0:np.size(glob_tpr_keras1), 3] = glob_tpr_keras1
          universal_tpr_keras[0:np.size(glob_tpr_keras2), 4] = glob_tpr_keras2
          universal_tpr_keras[0:np.size(glob_tpr_keras3), 5] = glob_tpr_keras3


          universal_auc[:, 3] = glob_auc1
          universal_auc[:, 4] = glob_auc2
          universal_auc[:, 5] = glob_auc3

        if b == 2:

          universal_fpr_keras[0:np.size(glob_fpr_keras1), 6] = glob_fpr_keras1
          universal_fpr_keras[0:np.size(glob_fpr_keras2), 7] = glob_fpr_keras2
          universal_fpr_keras[0:np.size(glob_fpr_keras3), 8] = glob_fpr_keras3

          universal_tpr_keras[0:np.size(glob_tpr_keras1), 6] = glob_tpr_keras1
          universal_tpr_keras[0:np.size(glob_tpr_keras2), 7] = glob_tpr_keras2
          universal_tpr_keras[0:np.size(glob_tpr_keras3), 8] = glob_tpr_keras3


          universal_auc[:, 6] = glob_auc1
          universal_auc[:, 7] = glob_auc2
          universal_auc[:, 8] = glob_auc3

        if b == 3:

          universal_fpr_keras[0:np.size(glob_fpr_keras1), 9] = glob_fpr_keras1
          universal_fpr_keras[0:np.size(glob_fpr_keras2), 10] = glob_fpr_keras2
          universal_fpr_keras[0:np.size(glob_fpr_keras3), 11] = glob_fpr_keras3

          universal_tpr_keras[0:np.size(glob_tpr_keras1), 9] = glob_tpr_keras1
          universal_tpr_keras[0:np.size(glob_tpr_keras2), 10] = glob_tpr_keras2
          universal_tpr_keras[0:np.size(glob_tpr_keras3), 11] = glob_tpr_keras3


          universal_auc[:, 9] = glob_auc1
          universal_auc[:, 10] = glob_auc2
          universal_auc[:, 11] = glob_auc3

        if b == 4:

          universal_fpr_keras[0:np.size(glob_fpr_keras1), 12] = glob_fpr_keras1
          universal_fpr_keras[0:np.size(glob_fpr_keras2), 13] = glob_fpr_keras2
          universal_fpr_keras[0:np.size(glob_fpr_keras3), 14] = glob_fpr_keras3

          universal_tpr_keras[0:np.size(glob_tpr_keras1), 12] = glob_tpr_keras1
          universal_tpr_keras[0:np.size(glob_tpr_keras2), 13] = glob_tpr_keras2
          universal_tpr_keras[0:np.size(glob_tpr_keras3), 14] = glob_tpr_keras3


          universal_auc[:, 12] = glob_auc1
          universal_auc[:, 13] = glob_auc2
          universal_auc[:, 14] = glob_auc3
