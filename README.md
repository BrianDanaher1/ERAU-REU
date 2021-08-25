# Variable Stride Project

The purpose of this research was to evaluate the Variable Stride algorithm, a subsampling algorithm, as used in convolutional neural networks. Subsampling is a common technique used in CNNs in order to increase the training speed of the dataset at the expense of coarsening the data and decreasing the amount of useful information. This algorithm allows the user to section the image and determine the pooling stride, the size of the downsampling filter, in each section. By manipulating these parameters, one can isolate the portions of an image which are the most important in classification and only minimally downsample these regions, while more aggressively downsizing the less important parts of the image. 


This research was done in collaboration with the Signal Processing and Applied Mathematics group of the Nevada National Security Site (NNSS), and is a continuation of research performed in Spring 2021 as part of a mathematics capstone project. A team of three worked on this project: myself, Matthew Brown, and Ying Zheng. 

The experiment developed for this research quantified the performance of the Varibale Stride algorithm in a binary classification situation, and compared the algorithm to avgpool and maxpool. The data witch was used to evaluate the algorithm was a set of 200 retinal images taken from a publicly available diabetic retinopathy dataset. Diabetic retinopathy is a disease which affects diabetics, and can be identified from retinal images as the disease causes scarring and other abnormalities. While the initial dataset is sorted into 4 levels of severity of the disease, the NNSS’ pre-processed dataset is taken from two categories: healthy, and most severe. The resulting dataset was split evenly into 100 images per category, and the images have been pre-processed to be of a consistent 600x600 initial size. A blue filter has also been applied to increase training accuracy.



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
    
3. ttests.m
    Takes in an excel sheet of data and outputs a matrix of t-tests of the columns of data in all combinations.


## Neural Networks

The three CNN structures used in this experiment are displayed below. These structures were chosen by procedurally creating a series of networks with different structures and splitting the networks into groups based on filter/node count. The network structure with the highest validation loss in each group were selected (16, 32, and 64 nodes/filters).
and choosing the ones with the highest validation loss. 

![Networks](https://github.com/BrianDanaher1/ERAU-REU/blob/main/Network%20Structures.png?raw=true)

3 different variable stride schemes were created for this experiment: Variable Stride right (VSr), Variable Stride center (VSc), and Variable Stride custom (VScu). These schemes were created holistically by examining the activation maps from preliminary runs and developing the schemes to capture the ares most often highlighted in these maps. Multiple schemes were created to reduce the risk of a poorly-chosen striding scheme skewing the results. 

60 neural network runs were performed by each of the three researchers to spread out the computation time: 4 runs for each combination of the 5 pooling methods (VSr, VSc, VSCu, maxpool, and avgpool) and the 3 netowrk structures. 12 runs were performed in total for each combination of network structure and striding scheme. Within each run, training and testing losses and accuracies, along with the true and false positive rates were collected at the end of each epoch. 

## Analysis

The data analysis of this project is ongoing. Preliminary results analyzing the AUC curve suggest that the variable stride algorithm is not suited for diabetic retinopathy classification.
