# Fruit-Veggie-Classification

## Goal:

## Description:
Utilizing different CNN architectures and comparing their performances across 3 datasets. The datasets involved are duplicates of each other, but separated into different folders so they can be labeled differently. 
The number of classes across the datasets are as follows: Fruits_Vegetables_Dataset - 20 classes, Fruits_and_Veggies - 10 classes, Freshness - 2 classes.
The datasets were created and read into the code by reading from the folders they're contained in and labeled according to the folder name as opposed to the file name. Additional information of how the folders are structured is below.

After the datasets were created and properly labeled, the 5 different CNN architectures were created. We tested a variety of differnt architectures but settled on these 5: Single Convolution Layer, LeNet-5, VGG-16, Alexnet, and InceptionResNetV2. Various other architectures were tested, like ResNet50 and GoogLeNet, but these were either significantly outperformed or were too computationally expensive to run on the resources available to us.

After selecting our 5 archiectures, we hyperparameter tuned with Weights and Biases sweeps with 10 epochs per sweep and 5 sweeps for each model with random search algorithm to find an estimate for the best hyperparameters. We tested 3 for each model: learning rate, dropout and batch size. Due to hardware constraints and time limitations, we were only able to test these hyperparameters with the data that was classified by both food type and freshness. We used these same paramters on the model for the other datasets assuming minial differences.

Afterwards the models were trained and tested for 15 epochs using the values we obtained from hyperparameter tuning and we graphed out the accuracy and validation during training, and ultimately tested the 

This repository contains the notebook and the accompanying images that were obtained from our experiemnts. Is it currently lacking the dataset that was used because the file size is too large, but the kaggle link and labeling convention used will be attached towards the bottom.

In terms of variable naming conventions in order to differentiate the different models trained on which dataset, number 1 refers to the Fruits_Vegetables_Dataset with 20 classes, number 2 represents Fruits_and_Veggies, and number 3 represents freshness. The results from the experiments are as follows (only linking 2 of the 5 architectures for the sake of space and most experiments yielded similar results):

# Hyperparameter tuning:

LeNet-5:<br>
![alt text](https://github.com/etruong249/Fruit-Veggie-Classification/blob/main/ReadMe_images/lenet_tune.png)<br>

InceptionResNetV2:<br>
![alt text](https://github.com/etruong249/Fruit-Veggie-Classification/blob/main/ReadMe_images/Inceptionv2_tune.png)<br>

AlexNet:<br>
![alt text](https://github.com/etruong249/Fruit-Veggie-Classification/blob/main/ReadMe_images/alexnet_tune.png)<br>

VGG16:<br>
![alt text](https://github.com/etruong249/Fruit-Veggie-Classification/blob/main/ReadMe_images/vgg16_tune.png)<br>

SingleCNN:<br>
![alt text](https://github.com/etruong249/Fruit-Veggie-Classification/blob/main/ReadMe_images/singleCNN_tune.png)<br>


# Test Runs:
LeNet-5:<br>
Both freshness and fruit/veggie classification:<br>

![alt text](https://github.com/etruong249/Fruit-Veggie-Classification/blob/main/ReadMe_images/LeNet1-acc.png)<br>
![alt text](https://github.com/etruong249/Fruit-Veggie-Classification/blob/main/ReadMe_images/LeNet1-loss.png)<br>

Just fruit/veggie classification:<br>
![alt text](https://github.com/etruong249/Fruit-Veggie-Classification/blob/main/ReadMe_images/LeNet2-acc.png)<br>
![alt text](https://github.com/etruong249/Fruit-Veggie-Classification/blob/main/ReadMe_images/LeNet2-loss.png)<br>

Just freshness classification:<br>
![alt text](https://github.com/etruong249/Fruit-Veggie-Classification/blob/main/ReadMe_images/LeNet3-acc.png)<br>
![alt text](https://github.com/etruong249/Fruit-Veggie-Classification/blob/main/ReadMe_images/LeNet3-loss.png)<br>

InceptionResNetV2:<br>
Both freshness and fruit/veggie classification:<br>
![alt text](https://github.com/etruong249/Fruit-Veggie-Classification/blob/main/ReadMe_images/ResV2-1-acc.png)<br>
![alt text](https://github.com/etruong249/Fruit-Veggie-Classification/blob/main/ReadMe_images/ResV2-1-loss.png)<br>

Just fruit/veggie classification:<br>
![alt text](https://github.com/etruong249/Fruit-Veggie-Classification/blob/main/ReadMe_images/ResV2-2-acc.png)<br>
![alt text](https://github.com/etruong249/Fruit-Veggie-Classification/blob/main/ReadMe_images/ResV2-2-loss.png)<br>

Just freshness classification:<br>
![alt text](https://github.com/etruong249/Fruit-Veggie-Classification/blob/main/ReadMe_images/ResV2-3-acc.png)<br>
![alt text](https://github.com/etruong249/Fruit-Veggie-Classification/blob/main/ReadMe_images/ResV2-3-loss.png)<br>


#Ultimately the best

The examples listed above include the LeNet-5 models and InceptionResNetV2. The reason behind these 2 choices are to showcase the performance differences between models from scratch and a pretrained model that used weights trained from imagenet. Between the 2 models, the LeNet-5 obtained a much lower accuracy score and much higher loss value compared to the InceptionResNetV2 model. This could be due to a list of factors, but the primary reasons are due to the initial starting weights and the depth and complexity difference between the 2 models. The use of the inception module alone is more complex and allows for better computations of each image and its accompanying weight that's calculated. 
However, the use of the inception module is not generalizable to the other pretrained model used, which was the VGG-16. The VGG-16 is also pretrained using weights from imagenet, but performed substantially worse compared to the inceptionV2 architecture. It's performance still ranges about 10-15% higher in accuracy compared to LeNet-5. The use of pretrained models helps with computation times and memory usage, but it also allows for superior performance due to the generalizability of the pretrained data and just how generalizable our current dataset is.

A similar trend can also be seen across the 3 different classification tasks. As the number of classes decreased, the shape of the validation accuracy across the epochs became more rigid as opposed to fluid. This could be due to the incraesed compexity and variance of the model, but for output. Because the models began with classifying 20 classes, there are more complex decision boundaries which will result in higher variance and a more complex accuracy line. 

Differences in test accuracy between all 3 classification data splits:<br>
![alt text](https://github.com/etruong249/Fruit-Veggie-Classification/blob/main/ReadMe_images/Accuracy_Between_Tasks.png)<br>

Differences in test loss between all 3 classification data splits:<br>
![alt text](https://github.com/etruong249/Fruit-Veggie-Classification/blob/main/ReadMe_images/Loss_Between_Tasks.png)<br>

Additionally, while the pretrained network may seem a bit more inconsistent at first, the jagged shapes within the plots actually are significantly smaller differences from the general trend compared to the model trained from scratch. Some of these performance deviations can just between a whole 5% and a jump in loss, whereas the pretrained model has differences nearly 1/10th of the difference.

VGG16 is pretrained while LeNet is not:<br>
![alt text](https://github.com/etruong249/Fruit-Veggie-Classification/blob/main/ReadMe_images/LeNet_and_VGG16.png)<br>

