# **Behavioral Cloning** 

---

[//]: # (Image References)

[image1]: ./examples/road.png " Image of center lane driving" 
[image2]: ./examples/Le-Net.jpg "LeNet-5"
[image3]: ./examples/output.gif "Output"
[image4]: ./examples/simulator.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Normal Image"
[image6]: ./examples/placeholder_small.png "Flipped Image"

## Introduction

This repository contains code for a project I did as a part of Udacity's Self Driving Car Nanodegree Program. I train a car to drive itself in a simulator. Here I apply the concepts of Deep Learning and Convolutional Neural Networks to teach the computer to drive car autonomously. This project makes use of a Udacity-developed driving simulator and training data collected from the simulator (neither of which is included in this repo). 
We feed the data collected from Simulator to our model, this data is fed in the form of images captured by 3 dashboard cams center, left and right. The output data contains a file data.csv which has the mappings of center, left and right images and the corresponding steering angle, throttle, brake and speed.
 The challenge in this project is to collect all sorts of training data so as to train the model to respond correctly in any type of situation.
 
 ![alt text][image4]

#### Video of the camera mounted on top of the car (driving itself)

![alt text][image3]

---

## Overview

The [goals / steps](https://review.udacity.com/#!/rubrics/432/view) of this project are the following:
* Using the simulator to collect data of good driving behavior
* Building, a convolution neural network in Keras that predicts steering angles from images
* Training and validating the model with a training and validation set
* Testing that the model successfully drives around track one without leaving the road

### Project Files
|  Filename   |   Description  | 
|:-------------:|:-------------:|
| model.py | define and train the neual network |
| model.h5 | saved model by keras |
| drive.py | communicate with simulator and use saved model to predict steering angle  |

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start of with a simple approach and then work my way up.

My first step was to use a convolution neural network model similar to the LeNet-5 model. I thought this model might be appropriate because it was simple and I understood it. I added a few Dropout layers and a Lambder layer and trained the model.

![alt text][image2]

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I increased the number of dropout layers and reduced the Epochs. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. It was hard for the car to recover from the side of the road, back to the center. Therefore, I used the left and right camera images to simulate the effect of car wandering off to the side, and recovering. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 74-100) consisted of a convolution neural network with the following layers and layer sizes:
```sh

model.add(Lambda(lambda x: x / 255.0 - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

#layer 1: Convolution, filters: 24, filter size: 5x5
model.add(Conv2D(filters=24, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D())

#layer 2: Convolution, filters: 36, filter size: 5x5
model.add(Conv2D(filters=36, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D())

#layer 3: Convolution, filters: 63, filter size: 3x3
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D())

#layer 4: Convolution, filters: 64, filter size: 3x3
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

#layer 5: fully connected layer
model.add(Dense(units=120, activation='relu'))

#Adding a dropout layer to avoid overfitting.
model.add(Dropout(0.2))

#layer 6: fully connected layer
model.add(Dense(units=84, activation='relu'))

#layer 7: fully connected layer
model.add(Dense(units=10, activation='relu'))

#layer 8: fully connected layer
model.add(Dense(units=1))

```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get on track to the middle again. Hereare some example image:

![alt text][image1]

To augment the data sat, I also flipped images and angles thinking that this would reduce the chance of the data set beeing biase. Since the test road is a circle, the car might always tend to drive to the left. 

After the collection process, I had 94147 number of data points. It took a while to train the network.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The  number of epochs was 3 as the val_loss and training_loss which were both decreasing after each epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.
