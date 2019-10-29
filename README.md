# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

This repository contains code for a project I did as a part of Udacity's Self Driving Car Nano Degree Program. I trained a car to drive itself in a simulator. The car was trained to drive itself using a deep neural network. 

The goals / steps of this project are the following:
* Using the simulator to collect data of good driving behavior
* Building, a convolution neural network in Keras that predicts steering angles from images
* Training and validating the model with a training and validation set
* Testing that the model successfully drives around track one without leaving the road

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Image of center lane driving" 
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start of with a simple approach and then work my way up.

My first step was to use a convolution neural network model similar to the LeNet-5 model. I thought this model might be appropriate because it was simple and I understood it. I added a few Dropout layers and a Lambder layer and trained the model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I increased the number of dropout layers and reduced the Epochs. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. It was hard for the car to recover from the side of the road, back to the center. Therefore, I used the left and right camera images to simulate the effect of car wandering off to the side, and recovering. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:
```sh

NETWORK

```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get on track to the middle again. These images show what a recovery looks like starting from the side of the road:

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I also flipped images and angles thinking that this would reduce the chance of the data set beeing biase. Since the test road is a circle, the car might always tend to drive to the left. 

After the collection process, I had 94147 number of data points. It took a while to train the network.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The  number of epochs was 3 as evidenced by the val_loss and training_loss which were both decreasing after each epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.
