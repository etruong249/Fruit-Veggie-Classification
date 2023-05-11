# Fruit-Veggie-Classification

Description:
Utilizing different CNN architectures and comparing their performances across 3 datasets. The datasets involved are duplicates of each other, but separated into different folders so they can be labeled differently. 
The number of classes across the datasets are as follows: Fruits_Vegetables_Dataset - 20 classes, Fruits_and_Veggies - 10 classes, Freshness - 2 classes.
The datasets were created and read into the code by reading from the folders they're contained in and labeled according to the folder name as opposed to the file name. Additional information of how the folders are structured is below.

After the datasets were created and properly labeled, they 5 different CNN architectures were created. These are: Single Convolution Layer, LeNet-5, VGG-16, Alexnet, and InceptionResNetV2.
Various other architectures were tested, like ResNet50 and GoogLeNet, but these were either significantly outperformed or were too computationally expensive to run on the resources available to us.

This repository contains the notebook and the accompanying images that were obtained from our experiemnts. Is it currently lacking the dataset that was used because the file size is too large, but the kaggle link and labeling convention used will be attached towards the bottom.

In terms of variable naming conventions in order to differentiate the different models trained on which dataset, number 1 refers to the Fruits_Vegetables_Dataset with 20 classes, number 2 represents Fruits_and_Veggies, and number 3 represents freshness. The results from the experiments are as follows (only linking 2 of the 5 architectures for the sake of space and most experiments yielded similar results):

LeNet-5:
LeNet1.png
