# **Behavioral Cloning** #

---

## **Behavioral Cloning Project** ##

The goals / steps of this project that I followed are the following:

* Use given data of good driving behavior
* Augment given data to increase diversity of good driving images
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Record successful driving with video
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/summary.png "Model Summary"
[image2]: ./examples/rgb.jpg "RGB Image"
[image3]: ./examples/hue.png "Hue Image"
[image4]: ./examples/saturation.png "Saturation Image"
[image5]: ./examples/value.png "Value Image"
[image6]: ./examples/RGB_cube.png "RGB Cube"
[image7]: ./examples/HSV_cylinder.png "HSV Cylinder"
[image8]: ./examples/normal.jpg "Normal Image"
[image9]: ./examples/flip.jpg "Flipped Image"


---

### Files Submitted ###

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode suitably modified for image preprocessing
* model.h5 containing a trained convolution neural network
* model.json containing a model structure
* video.mp4 a video of the car autonomously driving around the track 
* writeup_report.md summarizing the results

### How to Use Model ###

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

Next run the simulator and choose Autonomous Mode.


### Model Architecture and Training Strategy ###

####1. Solution Design Approach

My overall strategy was to utilize Convolutional Neural Network layers to process the input image.  This a natural step as CNN's have been state of the art at image classification since AlexNet in 2012.  Our
problem is one of regression (predicting the best continuous steering angle) rather than classification.
After flattening the resulting feature maps, the flattened vector is fed into a dense layer to a single
identity output neuron and trained with Mean Squared Error.  This will create a regression network rather
than classification (discrete number of classes) which would typically use softmax as an output layer.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 20% of the training data was used for validation.  To make it easier to train and capture
the best validation error model, I used a callback to automatically save the model.h5 (weights) file when
validation loss was lower that the previous minimum.

My training process was relatively fast as I chose to resize the original 160x320 images to 64x128.  I found
that the images fit into memory easily and a generator was not necessary.

I trained my first model after undergoing the preprocessing and augmentation of the dataset.  It worked 
reasonably well but drove off the track at a sharp turn.  On the second iteration, I increased the number
of 2D convolution filter from 3 to 6.  After retraining I found that the car could drive without any problems
around the track and never left the road.

The model is relatively small (around 7900 parameters) compared to example models given in lecture.  I believe
that I could easily shrink the model much farther (primarily by shrinking the image size) and train a 
perfectly acceptable autonomous driving network. 


#### Creation of the Training Set & Training Process

I found it difficult to use either the mouse or keyboard to generate appropriate center lane driving.  I relied
entirely on the given training data that I have preprocessed and augmented.

My major preprocessing is to convert the images from a standard RGB color space to an HSV color space.  HSV stands for Hue, Saturation, and Value and is commonly represented as a solid cylinder rather than a cube such as
RGB. The HSV color space is based on a perceptual representation of color.
The hue is basically color in general (the primary spectrum pure colors red, yellow, green, cyan, blue or magenta). It is common to represent the hue in a circle and give the value of the hue in degrees (360 degrees).
Saturation refers to the intensity of the color between gray (low saturation) and pure color (high saturation). 
The value corresponds to the brightness of a color, between black (low value) and average saturation (maximum value).

![alt text][image6]


Here we see the standard RGB cube representation.


![alt text][image7]

Above is the HSV cylinder color space.


Examples are given below for a standard RGB image:

![alt text][image2]

In the following we see (top to bottom images) the Hue, Saturation, and Value channel separately for this image.

![alt text][image3]
![alt text][image4]
![alt text][image5]





To augment the data sat, I also flipped images and took the negative steering angle in order to increase diversity as the test track is left turn biased. 

For example, here is a normal image that has then been flipped:

![alt text][image8]
![alt text][image9]

Finally, I incorporated the left and right side camera images along with a tuned correction factor for the
steering angle.

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### Model Architecture ####


My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24).

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####  Final Model Summary ####

Here is the output of Keras's model.summary() as a structured architecture summary. 

![alt text][image1]

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

