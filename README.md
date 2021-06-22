# **Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

![alt text][./img/predicted.png]



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/predstan/Traffic-Sign-Recogntion/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4044
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the each labels are present in the training data

![alt text][./img/visual.png]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. 

Pre-processing refers to techniques such as converting to grayscale, normalization, etc. In the project, I consider normalizing the images by dividing each pixel values by 255.0 so that each pixel values will be in the range of 0 and 1. It give the Model a robust and easier calculations when training and inference.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride-1x1,  outputs 29x29x64				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 27x27x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride-1x1,  outputs 26x26x128 			|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 24x24x256 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride=1x1,  outputs 23x23x256 			|
| Fully connected	    | weight = 256      							|
| Fully connected		| weights = 43       							|
| Softmax				| 43       										|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99%
* validation set accuracy of 98% 
* test set accuracy of 98%

I initially built and tried the Resnet Achitecture but Model was not converging so quickly, maybe due to computational power availbale on my computer. So i decided to use the LeNet architecture but model was overfitting.
For the Model that worked, I used 3 convolutional layer, and 2 fully connected layer and a Batch Normalization after each convolutional layer. I also used the dropout which helped prevent overfitting and model was able 
to achieve up to 98 percent on the Validation and training Set
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I downloaded all the images from the Germain dataset and Model achieved 99% on the whole dataset Corpus.

![alt text][./img/german_corpus.png] 



#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

The model is relatively sure about each signs predicted ( with probability ranging from 0.99 to 1.)




