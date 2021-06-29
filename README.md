# Variable Stride Project

The purpose of this research is to evaluate the Variable Stride Algorithm, a subsampling algorithm, as used in convolutional neural networks. Subsampling is a common technique used in CNNS in order to both increase the training speed of the dataset, with the tradeoff of coarsening the dataset and decreasing the amount of useful information. This algorithm allows the user to section the image and determine the pooling stride, the size of the downsampling filter, in each section. My manipulating these parameters, one can isolate the portions of an image which are most important in classification and only minimally downsample these regions, while more aggressively downsizing the less important parts of the image. 


This research is done in collaboration with the Signal Processing and Applied Mathematics group of the Nevada National Security Site (NNSS), and is a continuation of research performed in Spring 2021 as part of a mathematics capstone project. A team of three worked on this project: myself, Matthew Brown, and Ying Zheng. 

This experiment aims to quantify the performance of the Varibale Stride algorithm in a binary classification situation, and compare the algorithm to avgpool and maxpool. The data witch will be used to evaluate the algorithm is is a set of 200 retinal images taken from a publicly available diabetic retinopathy dataset. Diabetic retinopathy is a disease which affects diabetics, and can be identified from retinal images as the disease causes scarring and other abnormalities. While the initial dataset is sorted into 4 levels of severity of the disease, the NNSS’ pre-processed dataset is taken from two categories: healthy, and most severe. The resulting dataset is split evenly into 100 images per category, and the images have been pre-processed to be of a consistent 600x600 initial size. A blue filter has also been applied to increase training accuracy.



## List of Programs

Python Code

The majority of this project was done in Google Collab: an online Python library that allows block-by-block execution of code and real-time collaborative editing.

1. Preprocessing Code.py
    This code takes in the dataset of diabetic retinopathy images, and outputs .pickle files containing the images with maxpool, avgpool, and three variable stride implenetations applied. These .pickle files, along with a .pickle file containing the labels of the data, can subsequently be fed into neural networks. 

2. Main Neural Network Code.py 
    This code takes in the .pickle files, and uses them to run a total of 60 trials: the 5 pooling methods on three network structures four times each. The training loss and accuracy, the testing loss and accuracy, the classification reports, and the training times are collected for each run in both TensorBoard and plaintext. The true and false positive rates, as well as the AUCs of each run are outputted in excel files. 


MATLAB Code

These codes process the outputs of the main neural network code and organize the data into a usable format in excel.

1. Data Processing 1.m
    This code takes in the plaintext and outputs the accuracy, loss, validation accuracy, and validation loss in induvidual excel files.
    
2. Data Processing 2.m
    This code takes in the excel files from Data Processing 1.m as well as the excel sheets from Main Neural Network Code.py and formats them into one excel sheet.


## Neural Networks

The three CNN structures used in this experiment are displayed below. 

![poster2019](Network-Structures.png)
