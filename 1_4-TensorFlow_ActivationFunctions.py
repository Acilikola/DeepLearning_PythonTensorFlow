'''
Activation functions are a cornerstone of Machine Learning.
In general, Activation Functions define how a processing unit will treat its input --
usually passing this input through it and generating an output through its result
'''
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

# implement a basic function that plots a surface for an arbitrary activation function
# plot is done for all possible values of weight and bias between -0.5 and 0.5, with step 0.05
# input, weight and bias are 1-dimensional. input can also be passed as an argument
def plot_act(i=1.0, actfunc=lambda x: x):
    ws = np.arange(-0.5, 0.5, 0.05)
    bs = np.arange(-0.5, 0.5, 0.05)

    X, Y = np.meshgrid(ws, bs)

    os = np.array([actfunc(tf.constant(w*i + b)).eval(session=sess) \
                   for w,b in zip(np.ravel(X), np.ravel(Y))])

    Z = os.reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1)
    plt.show()

#dummy activation function
def func(x):
    return x

## BASIC STRUCTURE ##
#start a session
sess = tf.Session();
#create a simple input of 3 real values
i = tf.constant([1.0, 2.0, 3.0], shape=[1, 3])
#create a matrix of weights
w = tf.random_normal(shape=[3, 3])
#create a vector of biases
b = tf.random_normal(shape=[1, 3])
#tf.matmul will multiply the input(i) tensor and the weight(w) tensor then sum the result with the bias(b) tensor.
act = func(tf.matmul(i, w) + b)
#Evaluate the tensor to a numpy array
print(act.eval(session=sess))
print()

plot_act(1.0, func)
##

## STEP FUNCTIONS ##
'''
The Step function was the first one designed for Machine Learning algorithms.
It consists of a simple threshold function that varies the Y value from 0 to 1.
This function has been historically utilized for classification problems, like Logistic Regression with two classes

The Step Function simply functions as a limiter. Every input that goes through this function will be applied
to gets either assigned a value of 0 or 1. As such, it is easy to see how it can be handy in classification problems.

There are other variations of the Step Function such as the Rectangle Step and others, but those are seldom used.

Tensorflow dosen't have a Step Function
'''
##

## SIGMOID FUNCTIONS ##
'''
The next in line for Machine Learning problems is the family of the ever-present Sigmoid functions.
Sigmoid functions are called that due to their shape in the Cartesian plane, which resembles an "S" shape.

Sigmoid functions are very useful in the sense that they "squash" their given inputs into a bounded interval.
This is exceptionally handy when combining these functions with others such as the Step function.

Most of the Sigmoid functions you should find in applications will be the
Logistic, Arctangent, and Hyperbolic Tangent/Tanh functions

TanH is the most widely used function is Sigmoid family

Tensorflow doesn't have Arctangent Function
'''
# logistic regression #
# defined as f(x) = 1 / (1 + e^-x)
# a Sigmoid over the (0,1) interval
plot_act(1, tf.sigmoid)

act = tf.sigmoid(tf.matmul(i, w) + b)
print(act.eval(session=sess))
print()

# arctangent #
# defined as f(x) = tan^-1x
# a Sigmoid over -(-pi/2, pi/2) interval

# hyperbolic tangent (TanH) #
# defined as f(x) = 2/(1 + e^-2x) - 1
# a Sigmoid over (-1, 1) interval

plot_act(1, tf.tanh)

act = tf.tanh(tf.matmul(i, w) + b)
print(act.eval(session=sess))
print()
##

## LINEAR UNIT FUNCTIONS ##
'''
Linear Units are the next step in activation functions. They take concepts from both Step and Sigmoid functions
and behave within the best of the two types of functions. Linear Units in general tend to be variation of
what is called the Rectified Linear Unit, or ReLU for short.

The ReLU is a simple function which operates within the [0, infinity) interval.
For the entirety of the negative value domain, it returns a value of 0,
while on the positive value domain, it returns x for any f(x)

While it may seem counterintuitive to utilize a pseudo-linear function instead of something like Sigmoids,
ReLUs provide some benefits which might not be understood at first glance.
For example, during the initialization process of a Neural Network model, in which
weights are distributed at random for each unit, (1)ReLUs will only activate approximately only in 50% of the times
-- which saves some processing power. (2)the ReLU structure takes care of what is called the
Vanishing and Exploding Gradient problem by itself. (3)this kind of activation function is directly relatable to the
nervous system analogy of Neural Networks (this is called Biological Plausibility).

The ReLU structure has also has many variations optimized for certain applications, but those are implemented
on a case-by-case basis and therefore aren't in the scope of this notebook. If you want to know more, s
earch for Parametric Rectified Linear Units or maybe Exponential Linear Units
'''
plot_act(1, tf.nn.relu)

act = tf.nn.relu(tf.matmul(i, w) + b)
print(act.eval(session=sess))
print()
##
