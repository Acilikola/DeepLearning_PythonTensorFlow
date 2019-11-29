'''
In this tutorial, we first classify MNIST using a simple Multi-layer perceptron and then,
in the second part, we use deeplearning to improve the accuracy of our results.

MNIST is a: "database of handwritten digits that has a training set of 60,000 examples,
and a test set of 10,000 examples. It is a subset of a larger set available from NIST.
The digits have been size-normalized and centered in a fixed-size image"
'''

import tensorflow as tf
'''
since 'from tensorflow.examples.tutorials.mnist import input_data' is deprecated in newer versions of TF,
we had to download the old 'input_data.py', put it in the same folder as the project, import that, and use it
on the mnist dataset downloaded under MNIST_data folder

new version shows this as proper way (https://www.tensorflow.org/tutorials/quickstart/advanced) : 

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
...
...

'''
#from tensorflow.examples.tutorials.mnist import input_data
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# The one-hot = True argument only means that, in contrast to Binary representation, the labels will be
# presented in a way that to represent a number N, the Nth bit is 1 while the other bits are 0
'''
Understanding the imported data

The imported data can be divided as follow:

Training (mnist.train) >> Use the given dataset with inputs and related outputs for training of NN.
In our case, if you give an image that you know that represents a "nine", this set will tell
the neural network that we expect a "nine" as the output.
  - 55,000 data points
  - mnist.train.images for inputs
  - mnist.train.labels for outputs
  
Validation (mnist.validation) >> The same as training, but now the data is used to generate model properties
(classification error, for example) and from this, tune parameters like the
optimal number of hidden units or determine a stopping point for the back-propagation algorithm
  - 5,000 data points
  - mnist.validation.images for inputs
  - mnist.validation.labels for outputs
  
Test (mnist.test) >> the model does not have access to this informations prior to the testing phase.
It is used to evaluate the performance and accuracy of the model against "real life situations".
No further optimization beyond this point.
  - 10,000 data points
  - mnist.test.images for inputs
  - mnist.test.labels for outputs
'''
# create interactive session
sess = tf.InteractiveSession()

# create placeholders
'''
Placeholder 'X': represents the "space" allocated input or the images.
    - Each input has 784 pixels distributed by a 28 width x 28 height matrix
    - The 'shape' argument defines the tensor size by its dimensions.
    - 1st dimension = None. Indicates that the batch size, can be of any size.
    - 2nd dimension = 784. Indicates the number of pixels on a single flattened MNIST image.
    
Placeholder 'Y': represents the final output or the labels.
    - 10 possible classes (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    - The 'shape' argument defines the tensor size by its dimensions.
    - 1st dimension = None. Indicates that the batch size, can be of any size.
    - 2nd dimension = 10. Indicates the number of targets/outcomes
    
dtype for both placeholders: if you not sure, use tf.float32.
The limitation here is that the later presented softmax function only accepts float32 or float64 dtypes
'''
x  = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# assign bias and weights to null tensors
'''
values that we choose here can be critical, but we'll cover a better way on the second part,
instead of this type of initialization
'''
W = tf.Variable(tf.zeros([784, 10],tf.float32)) # Weight tensor
b = tf.Variable(tf.zeros([10],tf.float32)) # Bias tensor

# execute assignment operation
sess.run(tf.global_variables_initializer()) # run the op 'initialize_all_variables' using an interactive session

# adding weights and biases to input
tf.matmul(x,W) + b

# softmax regression
'''
Softmax is an activation function that is normally used in classification problems.
It generate the probabilities for the output

Logistic function output is used for the classification between two target classes 0/1.
Softmax function is generalized type of logistic function.
That is, Softmax can output a multiclass categorical probability distribution
'''
y = tf.nn.softmax(tf.matmul(x,W) + b)

# cost function
'''
minimize the difference between the right answers (labels) and estimated outputs by our network
'''
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 'gradient descent' optimization
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# training batches
'''
Train using minibatch Gradient Descent.

In practice, Batch Gradient Descent is not often used because is too computationally expensive.
The good part about this method is that you have the true gradient, but with the expensive computing task
of using the whole dataset in one time. Due to this problem, Neural Networks usually use minibatch to train
'''
#Load 50 training examples for each training iteration   
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# test
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
print("The final accuracy for the simple ANN model is: {} % ".format(acc) )

sess.close() #finish the session

print('\n!!!!! END OF FIRST PART !!!!!\n')
### How to improve our model? ###
'''
Several options as follow:
- Regularization of Neural Networks using DropConnect
- Multi-column Deep Neural Networks for Image Classification
- APAC: Augmented Pattern Classification with Neural Networks
- Simple Deep Neural Network with Dropout

In the next part we are going to explore the option:
Simple Deep Neural Network with Dropout (more than 1 hidden layer)
'''
###


### 2ND PART: DEEP LEARNING APPLIED ON MNIST ###
'''
In the first part, we learned how to use a simple ANN to classify MNIST.
Now we are going to expand our knowledge using a Deep Neural Network.

Architecture of our network is:

(Input) -> [batch_size, 28, 28, 1] >> Apply 32 filter of [5x5]
(Convolutional layer 1) -> [batch_size, 28, 28, 32]
(ReLU 1) -> [?, 28, 28, 32]
(Max pooling 1) -> [?, 14, 14, 32]
(Convolutional layer 2) -> [?, 14, 14, 64]
(ReLU 2) -> [?, 14, 14, 64]
(Max pooling 2) -> [?, 7, 7, 64]
[fully connected layer 3] -> [1x1024]
[ReLU 3] -> [1x1024]
[Drop out] -> [1x1024]
[fully connected layer 4] -> [1x10]
'''

#initialize
sess.close() # finish possible remaining session
sess = tf.InteractiveSession()
width = 28 # width of image in pixels
height = 28 # height of image in pixels
flat = width * height # number of pixels in one image
class_output = 10 # number of possible classifications for the problem

# create placeholders for input & output
x  = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])

# convert images of dataset to tensors
'''
The input image is 28 pixels by 28 pixels, 1 channel (grayscale).

In this case, the first dimension is the batch number of the image, and can be of any size (so we set it to -1).
The second and third dimensions are width and height, and the last one is the image channels
'''
x_image = tf.reshape(x, [-1,28,28,1])  
print(x_image)

print()

## Convolutional Layer 1
# defining kernel weight and bias
'''
The Size of the filter/kernel is 5x5; Input channels is 1 (grayscale); and we need 32 different feature maps
(here, 32 feature maps means 32 different filters are applied on each image).
So, the output of convolution layer would be 28x28x32).

In this step, we create a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
'''
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # need 32 biases for 32 outputs

# convolve with weight tensor and add biases
'''
To create convolutional layer, we use 'tf.nn.conv2d'.
It computes a 2-D convolution given 4-D input and filter tensors.

Inputs:
    - tensor of shape [batch, in_height, in_width, in_channels]. x of shape [batch_size,28 ,28, 1]
    - a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels].
    W is of size [5, 5, 1, 32]
    - stride which is [1, 1, 1, 1]. The convolutional layer, slides the "kernel window" across the input tensor.
    As the input tensor has 4 dimensions: [batch, height, width, channels], then the convolution operates on a
    2D window on the height and width dimensions.
    strides determines how much the window shifts by in each of the dimensions.
    As the first and last dimensions are related to batch and channels, we set the stride to 1.
    But for second and third dimension, we could set other values, e.g. [1, 2, 2, 1]
    
Process:
    - Change the filter to a 2-D matrix with shape [5*5*1,32]
    - Extracts image patches from the input tensor to form a virtual tensor of shape [batch, 28, 28, 5*5*1].
    - For each batch, right-multiplies the filter matrix and the image vector.
    
Output:
    A Tensor (a 2-D convolution) of size tf.Tensor 'add_7:0' shape=(?, 28, 28, 32)

Notice: the output of the first convolution layer is 32 [28x28] images.
Here 32 is considered as volume/depth of the output image.
'''
convolve1= tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1

# Apply the ReLU activation Function
'''
go through all outputs convolution layer, convolve1, and wherever a negative number occurs, we swap it out for a 0.
It is called ReLU activation Function.
f(x) = max(0,x)
'''
h_conv1 = tf.nn.relu(convolve1)

# Apply the max pooling
'''
max pooling is a form of non-linear down-sampling. It partitions the input image into a set of rectangles
and then find the maximum value for that region.

Lets use 'tf.nn.max_pool' function to perform max pooling.
    - Kernel size: 2x2 (if the window is a 2x2 matrix, it would result in one output pixel)
    - Strides: dictates the sliding behaviour of the kernel. In this case it will move 2 pixels everytime,
    thus not overlapping. The input is a matrix of size 28x28x32, and the output would be a matrix of size 14x14x32
'''
conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2
print(conv1)

print()
##
## Convolution layer 2
# defining kernel weight and bias
'''
We apply the convolution again in this layer.

    - Filter/kernel: 5x5 (25 pixels)
    - Input channels: 32 (from the 1st Conv layer, we had 32 feature maps)
    - 64 output feature maps
    
Notice: here, the input image is [14x14x32], the filter is [5x5x32], we use 64 filters of size [5x5x32],
and the output of the convolutional layer would be 64 convolved image, [14x14x64].

Notice: the convolution result of applying a filter of size [5x5x32] on image of size [14x14x32] is
an image of size [14x14x1], that is, the convolution is functioning on volume
'''
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) #need 64 biases for 64 outputs

# Convolve image with weight tensor and add biases
convolve2= tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2

# Apply the ReLU activation Function
h_conv2 = tf.nn.relu(convolve2)

# Apply the max pooling
conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2
print(conv2)

print()
##
## Fully Connected Layer
'''
You need a fully connected layer to use the Softmax and create the probabilities in the end.
Fully connected layers take the high-level filtered images from previous layer, that is all 64 matrices,
and convert them to a flat array.

So, each matrix [7x7] will be converted to a matrix of [49x1], and then all of the 64 matrix will be connected,
which make an array of size [3136x1]. We will connect it into another layer of size [1024x1].
So, the weight between these 2 layers will be [3136x1024]
'''
# flattening second layer
layer2_matrix = tf.reshape(conv2, [-1, 7 * 7 * 64])

# Weights and Biases between layer 2 and 3
'''
1)Composition of the feature map from the last layer (7x7) multiplied by the number of feature maps (64)
2)1027 outputs to Softmax layer
'''
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024])) # need 1024 biases for 1024 outputs

# Matrix Multiplication (applying weights and biases)
fcl = tf.matmul(layer2_matrix, W_fc1) + b_fc1

# Apply the ReLU activation function
h_fc1 = tf.nn.relu(fcl)
print(h_fc1)

print()
##
## Dropout Layer, Optional phase for reducing overfitting
'''
It is a phase where the network "forget" some features.

At each training step in a mini-batch, some units get switched off randomly so that it will not interact with
the network. That is, it weights cannot be updated, nor affect the learning of the other network nodes.
This can be very useful for very large neural networks to prevent overfitting
'''
keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)
print(layer_drop)

print()
##
## Readout Layer (Softmax Layer)
'''
Type: Softmax, Fully Connected Layer
'''
# Weights and Biases
'''
In last layer, CNN takes the high-level filtered images and translate them into votes using softmax.
Input channels: 1024 (neurons from the 3rd Layer); 10 output features
'''
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1)) #1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10])) # 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]

# Matrix Multiplication (applying weights and biases)
fc=tf.matmul(layer_drop, W_fc2) + b_fc2

# Apply the Softmax activation Function
'''
softmax allows us to interpret the outputs of fcl4 as probabilities. So, y_conv is a tensor of probabilities
'''
y_CNN= tf.nn.softmax(fc)
print(y_CNN)

print()
##

# DEFINE LOSS FUNCTION
'''
We need to compare our output, layer4 tensor, with ground truth for all mini_batch.
we can use cross entropy to see how bad our CNN is working - to measure the error at a softmax layer.

'reduce_sum' computes the sum of elements of (y_ * tf.log(layer4) across second dimension of the tensor,
and 'reduce_mean' computes the mean of all elements in the tensor

The following code shows an toy sample of cross-entropy for a mini-batch of size 2 which its items have been classified.
You can run it to see how cross entropy changes:

---
import numpy as np
layer4_test =[[0.9, 0.1, 0.1],[0.9, 0.1, 0.1]]
y_test=[[1.0, 0.0, 0.0],[1.0, 0.0, 0.0]]
np.mean( -np.sum(y_test * np.log(layer4_test),1))
---
'''
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]))

# DEFINE THE OPTIMIZER
'''
we want minimize the error of our network which is calculated by cross_entropy metric.
To solve the problem, we have to compute gradients for the loss (which is minimizing the cross-entropy)
 and apply gradients to variables. It will be done by an optimizer: GradientDescent or Adagrad
'''
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# DEFINE THE PREDICTION
'''
know how many of the cases in a mini-batch has been classified correctly; lets count them
'''
correct_prediction = tf.equal(tf.argmax(y_CNN, 1), tf.argmax(y_, 1))

# DEFINE ACCURACY
'''
report accuracy using average of correct cases
'''
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Run session, train
sess.run(tf.global_variables_initializer())

# fast result version
for i in range(1100):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, float(train_accuracy)))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print()
# actual result version <uncomment>
'''
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
'''

## EVALUATE THE MODEL
# print evaluation to user
n_batches = mnist.test.images.shape[0] // 50 # evaluate in batches to avoid out-of-memory issues
cumulative_accuracy = 0.0
for index in range(n_batches):
    batch = mnist.test.next_batch(50)
    cumulative_accuracy += accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
print("test accuracy {}".format(cumulative_accuracy / n_batches))

print()

# visualization
'''
take a look at all the filters
'''
kernels = sess.run(tf.reshape(tf.transpose(W_conv1, perm=[2, 3, 0,1]),[32, -1]))

import utils1 # downloaded in same folder from: http://deeplearning.net/tutorial/code/utils.py
from utils1 import tile_raster_images
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image = Image.fromarray(tile_raster_images(kernels, img_shape=(5, 5) ,tile_shape=(4, 8), tile_spacing=(1, 1)))
plt.rcParams['figure.figsize'] = (18.0, 18.0)
imgplot = plt.imshow(image)
imgplot.set_cmap('gray')
plt.show()


plt.rcParams['figure.figsize'] = (5.0, 5.0)
sampleimage = mnist.test.images[1]
plt.imshow(np.reshape(sampleimage,[28,28]), cmap="gray")
plt.show()

'''
output of an image through the first convolution layer
'''
ActivatedUnits = sess.run(convolve1,feed_dict={x:np.reshape(sampleimage,[1,784],order='F'),keep_prob:1.0})
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20,20))
n_columns = 6
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0,:,:,i], interpolation="nearest", cmap="gray")
plt.show()
'''
output of an image through the second convolution layer
'''
ActivatedUnits = sess.run(convolve2,feed_dict={x:np.reshape(sampleimage,[1,784],order='F'),keep_prob:1.0})
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20,20))
n_columns = 8
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0,:,:,i], interpolation="nearest", cmap="gray")
plt.show()
##
sess.close() #finish the session
###
