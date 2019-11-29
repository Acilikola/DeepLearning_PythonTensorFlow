'''
In this lesson, we will learn more about the key concepts behind the CNNs (Convolutional Neural Networks from now on)
'''
import numpy as np

## Convolution: 1D operation with Python (Numpy/Scipy)
'''
1D Convolution equation ==> y[n] = SUM_over_k{x[k].h[n-k]}

In this first example, we will use the pure mathematical notation.
Here we have a one dimensional convolution operation.
Lets say h is our image, i is index and x is our kernel:
x[i] = { 3, 4, 5 }
h[i] = { 2, 1, 0 }
'''
h = [2, 1, 0]
x = [3, 4, 5]
 
y = np.convolve(x, h)
print(y)
print()
'''
sliding x window over h:

6 = 2 * 3 : [[3 4 5][2 0 0]]
11 = 1 * 3 + 2 * 4 : [[3 4 5][1 2 0]]
14 = 0 * 3 + 1 * 4 + 2 * 5 : [[3 4 5][0 1 2]]
5 = 0 * 4 + 1 * 5 : [[3 4 5][0 0 1]]
0 = 0 * 5 : [[3 4 5][0 0 0]]
'''
print("Compare with the following values from Python: y[0] = {0} ; y[1] = {1}; y[2] = {2}; y[3] = {3}; y[4] = {4}".format(y[0], y[1], y[2], y[3], y[4])) 
print()

## 3 methods to apply kernel on the matrix, with padding (full), with padding(same) and without padding(valid)
# 'full' padding
'''
Lets think of the kernel as a sliding window. We have to come with the solution of padding zeros on the input array.

This is a very famous implementation and will be easier to show how it works with a simple example, consider this case:
x[i] = [6,2]
h[i] = [1,2,5,4]

Using the zero padding, we can calculate the convolution.
You have to invert the filter x, otherwise the operation would be cross-correlation.

First step, (now with zero padding): 2*0 + 6*1 = 6
[2  6]
 v  v
 0 [1 2 5 4]

Second step: 2*1 + 6*2 = 14
  [2 6]
   v v
0 [1 2 5 4]

Third step: 2*2 + 6*5 = 34
    [2 6]
     v v
0 [1 2 5 4]

Fourth step: 2*5 + 6*4 = 34
      [2 6]
       v v
0 [1 2 5 4]

Fifth step: 2*4 + 6*0 = 8
        [2  6]
         v  v
0 [1 2 5 4] 0
'''
x = [6, 2]
h = [1, 2, 5, 4]
y = np.convolve(x, h, "full") #now, because of the zero padding, the final dimension of the array is bigger
print(y)
print()

# 'same' padding
'''
In this approach, we just add the zero to left (and top of the matrix in 2D).
That is, only the first 4 steps of "full" method
'''
y = np.convolve(x, h, "same") #it is same as zero padding, but returns an ouput with the same length as max of x or h
print(y)
print()

# 'valid'/no padding
'''
In the last case we only applied the kernel when we had a compatible position on the h array,
in some cases you want a dimensionality reduction.
For this purpose, we simple ignore the steps that would need padding
'''
y = np.convolve(x, h, "valid") #valid returns output of length max(x, h) - min(x, h) + 1
                               #to ensure that values outside of the boundary of h will not be used in conv calculation
print(y)
print()
##
##

from scipy import signal as sg

## Convolution: 2D operation with Python (Numpy/Scipy)
'''
2D Convolution equation ==> I' = SUM_over_u_v{I(x-u,y-v)*g(u,v)}

Below we will apply the equation to an image represented by a 3x3 matrix according to the function g = (-1 1).
Please note that when we apply the kernel we always use its inversion.
    [255  7   3 ]
I = [212 240  4 ]
    [218 216 230]

g = [-1 1]

I[1,1] = 1 * 0 + (-1 * 255) = -255
I[1,2] = 1 * 255 + (-1 * 7) = 248
I[1,3] = 1 * 7 + (-1 * 3) = 4
I[2,1] = 1 * 0 + (-1 * 212) = -212
...
'''
# EX 1
I= [[255,   7,  3],
    [212, 240,  4],
    [218, 216, 230],]

g= [[-1, 1]]

print('Without zero padding (valid) \n')
print('{0} \n'.format(sg.convolve( I, g, 'valid')))
# The 'valid' argument states that the output consists only of those elements that do not rely on the zero-padding

print('With zero padding (full) \n')
print(sg.convolve( I, g))
print()

# EX 2

I= [[255,   7,  3],
    [212, 240,  4],
    [218, 216, 230],]

g= [[-1,  1],
    [ 2,  3],]

print ('With zero padding (full) \n')
print ('{0} \n'.format(sg.convolve( I, g, 'full')))
# The output is the full discrete linear convolution of the inputs. 
# It will use zero to complete the input matrix

print ('With  padding (same) \n')
print ('{0} \n'.format(sg.convolve( I, g, 'same')))
# The output is the full discrete linear convolution of the inputs. 
# It will use zero to complete the input matrix

print ('Without zero padding (valid) \n')
print (sg.convolve( I, g, 'valid'))
# The 'valid' argument states that the output consists only of those elements 
#that do not rely on the zero-padding.
print()
##

import tensorflow as tf

## CODING with TENSORFLOW ##
'''
Suppose that you have two tensors:

3x3 filter (4D tensor = [3,3,1,1] = [width, height, channels, number of filters])
10x10 image (4D tensor = [1,10,10,1] = [batch size, width, height, number of channels]

The output size for zero padding 'SAME' mode will be:
the same as input = 10x10

The output size without zero padding 'VALID' mode:
input size - kernel dimension + 1 = 10 -3 + 1 = 8 = 8x8
'''
input = tf.Variable(tf.random_normal([1, 10, 10, 1]))
filter = tf.Variable(tf.random_normal([3, 3, 1, 1]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

#Initialization and session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print("Input \n")
    print('{0} \n'.format(input.eval()))
    print("Filter/Kernel \n")
    print('{0} \n'.format(filter.eval()))
    print("Result/Feature Map with valid positions \n")
    result = sess.run(op)
    print(result)
    print('\n')
    print("Result/Feature Map with padding \n")
    result2 = sess.run(op2)
    print(result2)

print()
##

## CONVOLUTION APPLIED on IMAGES ##
'''
Upload your own image (drag and drop to folder) and type its name on the input field.
The result of this pre-processing will be an image with only a grayscale channel.

example: bird.jpg
'''
#Importing
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image #pip install image || pip install Pillow

im = Image.open('bird.jpg')  # type here your image's name

image_gr = im.convert("L")    # convert("L") translate color images into black and white
                              # uses the ITU-R 601-2 Luma transform (there are several 
                              # ways to convert an image to grey scale)
print("\n Original type: %r \n\n" % image_gr)

# convert image to a matrix with values from 0 to 255 (uint8) 
arr = np.asarray(image_gr) 
print("After conversion to numerical representation: \n\n %r" % arr) 

# Plot image
imgplot = plt.imshow(arr)
imgplot.set_cmap('gray')  #you can experiment different colormaps (Greys,winter,autumn)
print("\n Input image converted to gray scale: \n")
plt.show(imgplot)

### experiment with edge detector kernel
kernel = np.array([[ 0, 1, 0],
                   [ 1,-4, 1],
                   [ 0, 1, 0],]) 

grad = sg.convolve2d(arr, kernel, mode='same', boundary='symm')

print('GRADIENT MAGNITUDE - Feature map')
fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad), cmap='gray')
plt.show()

# change biases
'''
If we change the kernel and start to analyze the outputs we would be acting as a CNN.
The difference is that a Neural Network do all this work automatically (the kernel adjustment using different weights)
In addition, we can understand how biases affect the behaviour of feature maps

Please note that when you are dealing with most of the real applications of CNNs,
you usually convert the pixels values to a range from 0 to 1. This process is called normalization.
'''
type(grad)
grad_biases = np.absolute(grad) + 100
grad_biases[grad_biases > 255] = 255

print('GRADIENT MAGNITUDE - Feature map')
fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad_biases), cmap='gray')
plt.show()
###

### let's see how it works for a digit
'''
example: num3.jpg
'''
im = Image.open('num3.jpg')  # type here your image's name

image_gr = im.convert("L")    # convert("L") translate color images into black and white
                              # uses the ITU-R 601-2 Luma transform (there are several 
                              # ways to convert an image to grey scale)
print("\n Original type: %r \n\n" % image_gr)

# convert image to a matrix with values from 0 to 255 (uint8) 
arr = np.asarray(image_gr) 
print("After conversion to numerical representation: \n\n %r" % arr) 

# Plot image
fig, aux = plt.subplots(figsize=(10, 10))
imgplot = plt.imshow(arr)
imgplot.set_cmap('gray')  #you can experiment different colormaps (Greys,winter,autumn)
print("\n Input image converted to gray scale: \n")
plt.show(imgplot)

#### experiment with edge detector kernel
kernel = np.array([
                        [ 0, 1, 0],
                        [ 1,-4, 1],
                        [ 0, 1, 0],
                                     ]) 

grad = sg.convolve2d(arr, kernel, mode='same', boundary='symm')

print('GRADIENT MAGNITUDE - Feature map')
fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad), cmap='gray')
plt.show()
####
###
##
