#**Traffic Sign Recognition Writeup**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Files submitted
The following repository contains all the required files to review the submission
including HTML version of the notebook.

The results [can be viewed here](http://htmlpreview.github.io/?https://github.com/mateuszmrozewski/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)

## Dataset Exploration

### Dataset Summary
Dataset can be summarized in the following manner:
```
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```

### Exploratory visualization
For the visualization part I have displayed a histogram of classes:

![Histogram](./examples/class_histogram.png)

This is useful to see that not all of the classes have the same number training examples. It may
affect our predictions. For certain types of images.

Another visualization I did to display 50 random images. It is good to see some examples to get the idea
how to actual input looks like to see that some are darker, some are lighter, some were upsampled and some
were downsampled. You can see that images vary in quality which may affect the training process.

![Example images](./examples/signs.png)

## Design and Test Model Architeture

### Preprocessing
As data preprocessing I only transformed the images into grayscale and normalized the images. Initially I tried with all channels 
however I got slightly better results with grayscale.

### Model architecture
As a starting point I took the LeNet from the lab. It yielded pretty good results but not good enough to complete
the project. I decided to remove one of the fully connected layers as my intuition is that most of the 
important things happen on the convolutions. Following that lead I have added additional convolutional
layer and deepened the existing ones. It allowed to bump the accuracies to around 95%.
 
I experiment with dropout on both convolutional layers and fully connected layers but I only experienced
reduction of accuracy.

### Model training
I have used AdamOptimizer. Before changing the model (additional convolutional layer) I was experimenting 
with learning rate from 0.0001 to 0.001, epochs around 40 to 50 and batch sizes 64 and 128.

After adding the additional layer I bumped the learning rate up to 0.0008 and I was able to reduce number
of epochs to 20, as the model converged much quicker. 

### Solution approach
It was a trial-and-error approach with different shapes of the model and different values of hyperparameters.
Fortunately using GPU was fast enough for such approach.


## Test a Model on New Images
I have tested the model on a few images downloaded. I have to preprocess them the same way (resize to 32x32, 
grayscale and normalize). I got 100% accuracy on the first 5 images. One of the images had to be upscaled 
to match the 32x32 size which is a very promising result.
 
## Final thoughts
This project was an interesting follow up to the Deep Learning Nanodegree. I got a chance to practice more
what I have learnt previously. I know that I could push it further by trying to augment the training dataset
with additional variations of the images. However I was very satisfied with getting such a good result
with such a simple model.