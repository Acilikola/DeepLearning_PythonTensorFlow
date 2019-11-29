'''
Restricted Boltzmann Machine (RBM): RBMs are shallow neural nets that learn to reconstruct data by themselves
in an unsupervised fashion.

+ Why are RBMs important?
It can automatically extract meaningful features from a given input.

+ How does it work?
RBM is a 2 layer neural network. Simply, RBM takes the inputs and translates those into a set of binary values
that represents them in the hidden layer. Then, these numbers can be translated back to reconstruct the inputs.
Through several forward and backward passes, the RBM will be trained, and a trained RBM can reveal which features
are the most important ones when detecting patterns.

+ What are the applications of RBM?
RBM is useful for Collaborative Filtering, dimensionality reduction, classification, regression, feature learning,
topic modeling and even Deep Belief Networks.

+ Is RBM a generative or Discriminative model?
RBM is a generative model.

Let me explain it by first, see what is different between discriminative and generative models:
 - Discriminative: Consider a classification problem in which we want to learn to distinguish between Sedan cars
 (y = 1) and SUV cars (y = 0), based on some features of cars. Given a training set, an algorithm like
 logistic regression tries to find a straight line—that is, a decision boundary—that separates the suv and sedan.
 - Generative: looking at cars, we can build a model of what Sedan cars look like. Then, looking at SUVs,
 we can build a separate model of what SUV cars look like. Finally, to classify a new car, we can match the
 new car against the Sedan model, and match it against the SUV model, to see whether the new car looks
 more like the SUV or Sedan.

Generative Models specify a probability distribution over a dataset of input vectors. We can do both
supervised and unsupervised tasks with generative models:
 - In an unsupervised task, we try to form a model for P(x), where P is the probability given x as an input vector.
 - In the supervised task, we first form a model for P(x|y), where P is the probability of x given y(the label for x).
For example, if y = 0 indicates whether a car is a SUV or y = 1 indicates indicate a car is a Sedan,
then p(x|y = 0) models the distribution of SUVs’ features, and p(x|y = 1) models the distribution of Sedans’ features.
If we manage to find P(x|y) and P(y), then we can use Bayes rule to estimate P(y|x), because
P(y|x) = (p(x|y)*p(y))/p(x)
'''
import tensorflow as tf
import numpy as np
from PIL import Image
import utils1 # downloaded in same folder from: http://deeplearning.net/tutorial/code/utils.py
from utils1 import tile_raster_images
import matplotlib.pyplot as plt

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
import input_data

##RBM layers
'''
An RBM has two layers. The first layer of the RBM is called the visible (or input layer).
Imagine that our toy example, has only vectors with 7 values, so the visible layer must have j=7 input nodes.
The second layer is the hidden layer, which possesses i neurons in our case. Each hidden node can have
either 0 or 1 values (i.e., si = 1 or si = 0) with a probability that is a logistic function of the inputs it
receives from the other j visible units, called for example, p(si = 1). For our toy sample, we'll use 2 nodes
in the hidden layer, so i = 2

Each node in the first layer also has a bias. We will denote the bias as “v_bias” for the visible units.
The v_bias is shared among all visible units.

Here we define the bias of second layer as well. We will denote the bias as “h_bias” for the hidden units.
The h_bias is shared among all hidden units

We have to define weights among the input layer and hidden layer nodes. In the weight matrix,
the number of rows are equal to the input nodes, and the number of columns are equal to the output nodes.
Let W be the Tensor of 7x2 (7 - number of visible neurons, 2 - number of hidden neurons) that represents weights
between neurons.
'''
v_bias = tf.placeholder("float", [7])
h_bias = tf.placeholder("float", [2])

W = tf.constant(np.random.normal(loc=0.0, scale=1.0, size=(7, 2)).astype(np.float32))
##
## What RBM can do after training
'''
Think RBM as a model that has been trained based on images of a dataset of many SUV and Sedan cars.
Also, imagine that the RBM network has only two hidden nodes, one for the weight and, and one for the size of cars,
which in a sense, their different configurations represent different cars, one represent SUV cars and one for Sedan.
In a training process, through many forward and backward passes, RBM adjust its weights to send a stronger signal
to either the SUV node (0, 1) or the Sedan node (1, 0) in the hidden layer, given the pixels of images.
Now, given a SUV in hidden layer, which distribution of pixels should we expect? RBM can give you 2 things.
First, it encodes your images in hidden layer. Second, it gives you the probability of observing a case,
given some hidden values.

+ How to inference?
RBM has two phases:
 - Forward Pass
 - Backward Pass or Reconstruction
'''
# Phase 1 - Forward Pass
'''
+ Phase 1) Forward pass: Input one training sample (one image) X through all visible nodes, and pass it to
all hidden nodes. Processing happens in each node in the hidden layer. This computation begins by making
stochastic decisions about whether to transmit that input or not (i.e. to determine the state of each hidden layer).
At the hidden layer's nodes, X is multiplied by a W_ij and added to h_bias. The result of those two operations
is fed into the sigmoid function, which produces the node’s output, p(h_j), where j is the unit number.

    p(h_j) = logfunc(<SUM>{w_ij * x_i}) where logfunc = logistic function

p(h_j) represents the probabilities of the hidden units. And, all values together are called probability distribution.
That is, RBM uses inputs x to make predictions about hidden node activations. For example, imagine that the
values of p(h) for the first training item is [0.51 0.84]. It tells you what is the conditional probability for
each hidden neuron to be at Phase 1 ==> p(h_1 = 1|V) = 0.51 && p(h_2 = 1|V) = 0.84

As a result, for each row in the training set, a vector/tensor is generated, which in our case it is of size [1x2],
and totally n vectors p(h) = [nx2]

We then turn unit h_j on with probability p(h_j|V), and off with probability 1 - p(h_j|V)
herefore, the conditional probability of a configuration of h given v (for a training sample) is ==>
    p(h|V) = <MUL>{p(h_j|V)}

Now, sample a hidden activation vector h from this probability distribution p(h_j). That is, we sample the activation
vector from the probability distribution of hidden layer values.
'''
#let's look at a toy example for one case out of all input. Assume that we have a trained RBM, and a very simple
#input vector such as [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], lets see what would be the output of forward pass
sess = tf.Session()
X = tf.constant([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
v_state = X
print ("Input: ", sess.run(v_state))

h_bias = tf.constant([0.1, 0.1])
print ("hb: ", sess.run(h_bias))
print ("w: ", sess.run(W))

# Calculate the probabilities of turning the hidden units on:
h_prob = tf.nn.sigmoid(tf.matmul(v_state, W) + h_bias)  #probabilities of the hidden units
print ("p(h|v): ", sess.run(h_prob))

# Draw samples from the distribution:
h_state = tf.nn.relu(tf.sign(h_prob - tf.random_uniform(tf.shape(h_prob)))) #states
print ("h0 states:", sess.run(h_state))

print()
#
# Phase 2 - Backward Pass / Reconstruction
'''
+ Phase 2) Backward Pass (Reconstruction): The RBM reconstructs data by making several forward and backward passes
between the visible and hidden layers.

So, in the second phase (i.e. reconstruction phase), the samples from the hidden layer (i.e. h) play the role of input.
That is, h becomes the input in the backward pass. The same weight matrix and visible layer biases are used
to go through the sigmoid function. The produced output is a reconstruction which is an approximation of the
original input.
'''
vb = tf.constant([0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1])
print ("b: ", sess.run(vb))
v_prob = sess.run(tf.nn.sigmoid(tf.matmul(h_state, tf.transpose(W)) + vb))
print ("p(vi∣h): ", v_prob)
v_state = tf.nn.relu(tf.sign(v_prob - tf.random_uniform(tf.shape(v_prob))))
print ("v probability states: ", sess.run(v_state))

print()
#
# RBM
'''
RBM learns a probability distribution over the input, and then, after being trained, the RBM can generate
new samples from the learned probability distribution. As you know, probability distribution, is a mathematical
function that provides the probabilities of occurrence of different possible outcomes in an experiment.

The (conditional) probability distribution over the visible units v is given by
    p(v|h) = <MUL>{p(v_i|h)}
where p(v_i|h) = logfunc(a_i + <SUM>{w_ji * h_j})
'''
#given current state of hidden units and weights, what is the probability of generating [1. 0. 0. 1. 0. 0. 0.]
#in reconstruction phase, based on the above probability distribution function?
inp = sess.run(X)
print(inp)
print(v_prob[0])
v_probability = 1
for elm, p in zip(inp[0],v_prob[0]) :
    if elm ==1:
        v_probability *= p
    else:
        v_probability *= (1-p)
print(v_probability)

print()
'''
How similar X and V vectors are? Of course, the reconstructed values most likely will not look anything like
the input vector because our network has not trained yet. Our objective is to train the model in such a way that
the input vector and reconstructed vector to be same. Therefore, based on how different the input values look
to the ones that we just reconstructed, the weights are adjusted.
'''
#
##
print('!!!!!!!!!! MNIST !!!!!!!!!!')
print()
### MNIST
#The one-hot = True argument only means that, in contrast to Binary representation, the labels will be
#presented in a way that to represent a number N, the Nth bit is 1 while the other bits are 0
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

print(trX[1].shape) #look at dimension of the images

print()
'''
MNIST images have 784 pixels, so the visible layer must have 784 input nodes. For our case, we'll use 50 nodes
in the hidden layer, so i = 50

Let W be the Tensor of 784x50 (784 - number of visible neurons, 50 - number of hidden neurons) that represents
weights between the neurons
'''
vb = tf.placeholder("float", [784])
hb = tf.placeholder("float", [50])

W = tf.placeholder("float", [784, 50])

v0_state = tf.placeholder("float", [None, 784]) # define visible layer

# define hidden layer
h0_prob = tf.nn.sigmoid(tf.matmul(v0_state, W) + hb)  #probabilities of the hidden units
h0_state = tf.nn.relu(tf.sign(h0_prob - tf.random_uniform(tf.shape(h0_prob)))) #sample_h_given_X

# define reconstruction part
v1_prob = tf.nn.sigmoid(tf.matmul(h0_state, tf.transpose(W)) + vb) 
v1_state = tf.nn.relu(tf.sign(v1_prob - tf.random_uniform(tf.shape(v1_prob)))) #sample_v_given_h

'''
What is objective function?
 - Goal: Maximize the likelihood of our data being drawn from that distribution
 - Calculate error: In each epoch, we compute the "error" as a sum of the squared difference between step 1 and step n,
 e.g the error shows the difference between the data and its reconstruction.

Note: tf.reduce_mean computes the mean of elements across dimensions of a tensor
'''
# define error calculation
err = tf.reduce_mean(tf.square(v0_state - v1_state))

## How to train the model?
'''
The following part discuss how to train the model which needs some algebra background.

As mentioned, we want to give a high probability to the input data we train on. So, in order to train an RBM,
we have to maximize the product of probabilities assigned to all rows v (images) in the training set V
(a matrix, where each row of it is treated as a visible vector v):
    argmax(<MUL>{P(v)})
which is equivalent to maximazing the expected log probability of V:
    argmax(<SUM>{log(P(v))})

So, we have to update the weights w_ij to increase p(v) for all v in our training data during training. So we have to
calculate the derivative d[log(p(v))]\d[w_ij]

This cannot be easily done by typical gradient descent (SGD), so we can use another approach, which has 2 steps:
 - Gibbs Sampling
 - Contrastive Divergence
'''
# GIBBS SAMPLING
'''
Gibbs Sampling
First, given an input vector v we are using p(h|v) for prediction of the hidden values h
 - p(h|v) = sigmoid(X x W + h*b)
 - h0 = sampleProb(h0)
Then, knowing the hidden values, we use p(v|h) for reconstructing of new input values v
 - p(v|h) = sigmoid(h0 x transpose(W) + v*b)
 - v1 = sampleProb(v1) --> Sample v given h

This process is repeated k times. After k iterations we obtain an other input vector vk which was recreated from
original input values v0 or X.
Reconstruction steps:
 - Get one data point from data set, like x, and pass it through the net
 - Pass 0: (x)  ⇒  (h0)  ⇒  (v1) (v1 is reconstruction of the first pass)
 - Pass 1: (v1)  ⇒  (h1)  ⇒  (v2) (v2 is reconstruction of the second pass)
 - Pass 2: (v2)  ⇒  (h2)  ⇒  (v3) (v3 is reconstruction of the third pass)
 - Pass n: (vk)  ⇒  (hk+1)  ⇒  (vk+1)(vk is reconstruction of the nth pass)

+ What is sampling here (sampleProb)?
 - In forward pass: We randomly set the values of each h_i to be 1 with probability sigmoid(v x W + h*b)

To sample h given v ==> means to sample from the conditional probability distribution P(h|v).
It means that you are asking what are the probabilities of getting a specific set of values for the hidden neurons,
given the values v for the visible neurons, and sampling from this probability distribution.

 - In reconstruction: We randomly set the values of each v_i to be 1 with probability sigmoid(h x transpose(W) + v*b)
'''
# CONTRASTIVE DIVERGENCE (CD-k)
'''
The update of the weight matrix is done during the Contrastive Divergence step.

Vectors v0 and vk are used to calculate the activation probabilities for hidden values h0 and hk.
The difference between the outer products of those probabilities with input vectors v0 and vk results in
the update matrix: diff_W = v0 x h0 - vk x hk

Contrastive Divergence is actually matrix of values that is computed and used to adjust values of the W matrix.
Changing W incrementally leads to training of W values. Then on each step (epoch), W is updated to a new value W'
through the equation below ==>
    W' = W + alpha * diff_W

+ What is alpha?
Here, alpha is some small step rate and is also known as the "learning rate".
'''
# lets assume k = 1, that is we just get one more step
h1_prob = tf.nn.sigmoid(tf.matmul(v1_state, W) + hb)
h1_state = tf.nn.relu(tf.sign(h1_prob - tf.random_uniform(tf.shape(h1_prob)))) #sample_h_given_X

alpha = 0.01
W_Delta = tf.matmul(tf.transpose(v0_state), h0_prob) - tf.matmul(tf.transpose(v1_state), h1_prob)
update_w = W + alpha * W_Delta
update_vb = vb + alpha * tf.reduce_mean(v0_state - v1_state, 0)
update_hb = hb + alpha * tf.reduce_mean(h0_state - h1_state, 0)

# start sess and init variables
cur_w = np.zeros([784, 50], np.float32)
cur_vb = np.zeros([784], np.float32)
cur_hb = np.zeros([50], np.float32)
prv_w = np.zeros([784, 50], np.float32)
prv_vb = np.zeros([784], np.float32)
prv_hb = np.zeros([50], np.float32)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# look at error of first run
print('Error of first run: ', sess.run(err, feed_dict={v0_state: trX, W: prv_w, vb: prv_vb, hb: prv_hb}))

print()

# TRAIN
#Parameters
epochs = 5
batchsize = 100
weights = []
errors = []

for epoch in range(epochs):
    for start, end in zip( range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
        batch = trX[start:end]
        cur_w = sess.run(update_w, feed_dict={ v0_state: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_vb = sess.run(update_vb, feed_dict={v0_state: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_hb = sess.run(update_hb, feed_dict={ v0_state: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
        if start % 10000 == 0:
            errors.append(sess.run(err, feed_dict={v0_state: trX, W: cur_w, vb: cur_vb, hb: cur_hb}))
            weights.append(cur_w)
    print ('Epoch: %d' % epoch,'reconstruction error: %f' % errors[-1])
plt.plot(errors)
plt.xlabel("Batch Number")
plt.ylabel("Error")
plt.show()

print()

# final weight after training
uw = weights[-1].T
print(uw) # a weight matrix of shape (50,784)

print()
##

## LEARNED FEATURES
'''
We can take each hidden unit and visualize the connections between that hidden unit and each element in the
input vector. In our case, we have 50 hidden units. Lets visualize those.

Let's plot the current weights: tile_raster_images helps in generating an easy to grasp image from a set of
samples or weights. It transform the uw (with one flattened image per row of size 784), into an array (of size  25×20)
in which images are reshaped and laid out like tiles on a floor.
'''
tile_raster_images(X=cur_w.T, img_shape=(28, 28), tile_shape=(5, 10), tile_spacing=(1, 1))
image = Image.fromarray(tile_raster_images(X=cur_w.T, img_shape=(28, 28) ,tile_shape=(5, 10), tile_spacing=(1, 1)))
### Plot image
plt.rcParams['figure.figsize'] = (18.0, 18.0)
imgplot = plt.imshow(image)
imgplot.set_cmap('gray')
plt.show()

'''
Each tile in the above visualization corresponds to a vector of connections between a hidden unit and
visible layer's units.

Let's look at one of the learned weights corresponding to one of hidden units for example.

In this particular square, the gray color represents weight = 0, and the whiter it is, the more positive
the weights are (closer to 1). Conversely, the darker pixels are, the more negative the weights.
The positive pixels will increase the probability of activation in hidden units
(after multiplying by input/visible pixels), and negative pixels will decrease the probability of a unit
hidden to be 1 (activated). So, why is this important? So we can see that this specific square (hidden unit)
can detect a feature (e.g. a "/" shape) and if it exists in the input
'''
image = Image.fromarray(tile_raster_images(X =cur_w.T[10:11], img_shape=(28, 28),tile_shape=(1, 1), tile_spacing=(1, 1)))
### Plot image
plt.rcParams['figure.figsize'] = (4.0, 4.0)
imgplot = plt.imshow(image)
imgplot.set_cmap('gray')
plt.show()
##

print()
print('!!!!!!! RECONSTRUCT FIGURE 3 !!!!!!!')
print()
### RECONSTRUCT FIGURE 3 using our net
'''
Let's look at the reconstruction of an image now. Imagine that we have a destructed image of figure 3.

Lets see if our trained network can fix it:
'''
## plot the destructed figure 3 image (use destructed3.jpg under work directory)
img = Image.open('destructed3.jpg')
plt.rcParams['figure.figsize'] = (4.0, 4.0)
imgplot = plt.imshow(img)
imgplot.set_cmap('gray')
plt.show()
##

## pass this image through the net
#convert the image to a 1d numpy array
sample_case = np.array(img.convert('I').resize((28,28))).ravel().reshape((1, -1))/255.0
##

## feed the sample case into the network and construct the output
hh0_p = tf.nn.sigmoid(tf.matmul(v0_state, W) + hb)
#hh0_s = tf.nn.relu(tf.sign(hh0_p - tf.random_uniform(tf.shape(hh0_p)))) 
hh0_s = tf.round(hh0_p)
hh0_p_val,hh0_s_val  = sess.run((hh0_p, hh0_s), feed_dict={ v0_state: sample_case, W: prv_w, hb: prv_hb})
print("Probability nodes in hidden layer:" ,hh0_p_val)
print()
print("activated nodes in hidden layer:" ,hh0_s_val)
#reconstruct
vv1_p = tf.nn.sigmoid(tf.matmul(hh0_s_val, tf.transpose(W)) + vb)
rec_prob = sess.run(vv1_p, feed_dict={ hh0_s: hh0_s_val, W: prv_w, vb: prv_vb})

print()
##

## plot the reconstructed image
img = Image.fromarray(tile_raster_images(X=rec_prob, img_shape=(28, 28),tile_shape=(1, 1), tile_spacing=(1, 1)))
plt.rcParams['figure.figsize'] = (4.0, 4.0)
imgplot = plt.imshow(img)
imgplot.set_cmap('gray')
plt.show()
##
###
###
