# AUTOENCODERS
'''
An autoencoder, also known as autoassociator or Diabolo networks, is an artificial neural network
employed to recreate the given input. It takes a set of unlabeled inputs, encodes them and then tries to extract
the most valuable information from them. They are used for feature extraction, learning generative models of data,
dimensionality reduction and can be used for compression.

A 2006 paper named Reducing the Dimensionality of Data with Neural Networks, done by
G. E. Hinton and R. R. Salakhutdinov, showed better results than years of refining other types of network,
and was a breakthrough in the field of Neural Networks, a field that was "stagnant" for 10 years.

Now, autoencoders, based on Restricted Boltzmann Machines, are employed in some of the largest deep learning
applications. They are the building blocks of Deep Belief Networks (DBN).

+ An autoencoder can be divided in two parts, the encoder and the decoder.
 - The encoder needs to compress the representation of an input. In this case we are going to reduce the dimension
 the face of our actor, from 2000 dimensions to only 30 dimensions, by running the data through layers of our encoder.
 - The decoder works like encoder network in reverse. It works to recreate the input, as closely as possible.
 This plays an important role during training, because it forces the autoencoder to select the most important
 features in the compressed representation
'''

# Feature Extraction and Dimensionality Reduction
'''
An example given by Nikhil Buduma in KdNuggets
(https://www.kdnuggets.com/2015/03/deep-learning-curse-dimensionality-autoencoders.html) which gave an
excellent explanation of the utility of this type of Neural Network.

Say that you want to extract what emotion the person in a photography is feeling. Using the 256x256 pixel
grayscale picture as an example.

this image being 256x256 pixels in size correspond with an input vector of 65536 dimensions!
If we used an image produced with conventional cellphone cameras, that generates images of 4000 x 3000 pixels,
we would have 12 million dimensions to analyze.

According to a 1982 study by C.J. Stone, the time to fit a model
is optimal if ==>
m^(-p/(2*p + d)); m: # of data points; d: dimensionality of data; p: parameter that depends on model

Returning to our example, we don't need to use all of the 65536 dimensions to classify an emotion.
A human identify emotions according to some specific facial expression, some key features,
like the shape of the mouth and eyebrows
'''
# Training - Loss function
'''
An autoencoder uses the Loss function to properly train the network. The Loss function will calculate the
differences between our output and the expected results. After that, we can minimize this error with
gradient descent. There are more than one type of Loss function, it depends on the type of data

+ Binary Values ==> loss(f(x)) = -SUM{(x_k * log(xhat_k) + (1 - x_k)*log(1-xhat_k))}

For binary values, we can use an equation based on the sum of Bernoulli's cross-entropy.
x_k -> input, xhat_k -> corresponding output

+ Real Values ==> loss(f(x)) = -1/2 * SUM{(xhat_k - x_k)^2}

We can use the sum of squared differences for our Loss function for non-binary values. If you use this loss function,
it's necessary that you use a linear activation function for the output layer.
x_k -> input, xhat_k -> corresponding output

+ Loss Gradient ==> GRADIENT{loss(f(x^t))} = xhat^t - x^t

We use the gradient descent to reach the local minimum of our function loss(f(x^t)), taking steps towards the
negative of the gradient of the function in the current point
Our function about the gradient of the loss in the preactivation of the output layer.

It's actually a simple formula, it is done by calculating the difference between our output xhat_t and input x^t

Then our network backpropagates our gradient through the network using backpropagation.
'''
import tensorflow as tf
import numpy as np
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
## MNIST
#The one-hot = True argument only means that, in contrast to Binary representation, the labels will be
#presented in a way that to represent a number N, the Nth bit is 1 while the other bits are 0
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
##

## set parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}
##

## create encoder
'''
we are going to use sigmoidal functions. Sigmoidal functions delivers great results with this type of network.
This is due to having a good derivative that is well-suited to backpropagation
'''
def encoder(x):
    # Encoder first layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # Encoder second layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2

##

## create decoder
'''
You can see that the layer_1 in the encoder is the layer_2 in the decoder and vice-versa
'''
def decoder(x):
    # Decoder first layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1']))
    # Decoder second layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2

##

## construct model
'''
Let's construct our model. In the variable cost we have the loss function and in the optimizer variable
we have our gradient used for backpropagation
'''
# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Reconstructed Images
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()
##

## train model
'''
for training, run for 20 epochs
'''
sess = tf.InteractiveSession()
sess.run(init)

total_batch = int(mnist.train.num_examples / batch_size)
# Training cycle
for epoch in range(training_epochs):
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1),
              "cost=", "{:.9f}".format(c))

print("Optimization Finished!")

print()
##
# Applying encode and decode over test set
encode_decode = sess.run(
    y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})

# Compare original images with their reconstructions
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
plt.show()
'''
As you can see, the reconstructions were successful. It can be seen that some noise were added to the image
'''
