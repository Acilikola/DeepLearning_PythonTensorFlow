'''
How does TensorFlow work?

TensorFlow defines computations as Graphs, and these are made with operations (also know as “ops”).
So, when we work with TensorFlow, it is the same as defining a series of operations in a Graph.
To execute these operations as computations, we must launch the Graph into a Session.
The session translates and passes the operations represented into the graphs to the device
you want to execute them on, be it a GPU or CPU. In fact, TensorFlow's capability to execute
the code on different devices such as CPUs and GPUs is a consequence of it's specific structure.

For example, the image below represents a graph in TensorFlow.
W, x and b are tensors over the edges of this graph.
MatMul is an operation over the tensors W and x, after that
Add is called and add the result of the previous operator with b.
The resultant tensors of each operation cross the next one until the end where
it's possible to get the wanted result.
'''
import tensorflow as tf

# building a graph
graph1 = tf.Graph()
'''
Now we call the TensorFlow functions that construct new tf.Operation and tf.Tensor objects
and add them to the graph1. As mentioned, each tf.Operation is a node and each tf.Tensor is an edge in the graph.

Lets add 2 constants to our graph.
For example, calling tf.constant([2], name = 'constant_a') adds a single tf.Operation to the default graph.
This operation produces the value 2, and returns a tf.Tensor that represents the value of the constant.

Notice: tf.constant([2], name="constant_a") creates a new tf.Operation named "constant_a"
and returns a tf.Tensor named "constant_a:0"
'''
with graph1.as_default():
    a = tf.constant([2], name = 'constant_a')
    b = tf.constant([3], name = 'constant_b')

print(a)

# printing value of a
sess = tf.Session(graph = graph1)
result = sess.run(a)
print(result)
sess.close()

with graph1.as_default():
    c = tf.add(a,b) # c = a+b is also a way to define sum of terms

sess = tf.Session(graph = graph1)
result = sess.run(c)
print(result)
sess.close()

# avoid having to close sessions every time by defining them in a 'with' block; auto closes
with tf.Session(graph = graph1) as sess:
    result = sess.run(c)
    print(result)

# define multidimensional array
graph2 = tf.Graph()
with graph2.as_default():
    Scalar = tf.constant(2)
    Vector = tf.constant([5,6,2])
    Matrix = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
    Tensor = tf.constant([[[1,2,3],[2,3,4],[3,4,5]] , [[4,5,6],[5,6,7],[6,7,8]] , [[7,8,9],[8,9,10],[9,10,11]]])
with tf.Session(graph = graph2) as sess:
    result = sess.run(Scalar)
    print ("Scalar (1 entry):\n %s \n" % result)
    result = sess.run(Vector)
    print ("Vector (3 entries) :\n %s \n" % result)
    result = sess.run(Matrix)
    print ("Matrix (3x3 entries):\n %s \n" % result)
    result = sess.run(Tensor)
    print ("Tensor (3x3x3 entries) :\n %s \n" % result)

# tf.shape returns the shape of our data structure
print()
print(Scalar.shape)
print(Tensor.shape)

print()

# element wise multiplication (Hadamard product)
graph3 = tf.Graph()
with graph3.as_default():
    Matrix_one = tf.constant([[1,2,3],[2,3,4],[3,4,5]])
    Matrix_two = tf.constant([[2,2,2],[2,2,2],[2,2,2]])

    add_1_operation = tf.add(Matrix_one, Matrix_two)
    add_2_operation = Matrix_one + Matrix_two

with tf.Session(graph = graph3) as sess:
    result = sess.run(add_1_operation)
    print ("Defined using tensorflow function :")
    print(result)
    result = sess.run(add_2_operation)
    print ("Defined using normal expressions :")
    print(result)

print()

# matrix product
graph4 = tf.Graph()
with graph4.as_default():
    Matrix_one = tf.constant([[2,3],[3,4]])
    Matrix_two = tf.constant([[2,3],[3,4]])

    mul_operation = tf.matmul(Matrix_one, Matrix_two)

with tf.Session(graph = graph4) as sess:
    result = sess.run(mul_operation)
    print ("Defined using tensorflow function :")
    print(result)

print()

## VARIABLES ##
'''
TensorFlow variables are used to share and persistent some stats that are manipulated by our program.
That is, when you define a variable, TensorFlow adds a tf.Operation to your graph.
Then, this operation will store a writable tensor value that persists between tf.Session.run calls.
So, you can update the value of a variable through each run, while you cannot update tensor
(e.g a tensor created by tf.constant()) through multiple runs in a session.
'''
# define a variable
v = tf.Variable(0)
'''
Let's first create a simple counter, a variable that increases one unit at a time:

To do this we use the tf.assign(reference_variable, value_to_update) command.
tf.assign takes in two arguments, the reference_variable to update, and assign it to the value_to_update it by
'''
update = v.assign(v+1)
# Variables must be initialized by running an initialization operation after having launched the graph. We first have to add the initialization operation to the graph
init_op = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init_op)
    print(session.run(v))
    for _ in range(3):
        session.run(update)
        print(session.run(v))

print()
##

## PLACEHOLDERS ##
'''
If you want to feed data to a TensorFlow graph from outside a graph, you will need to use placeholders

Placeholders can be seen as "holes" in your model, "holes" which you will pass the data to
You can create them using tf.placeholder(datatype), where datatype specifies the type of data
(integers, floating points, strings, booleans) along with its precision (8, 16, 32, 64) bits

Data type	Python type	Description
DT_FLOAT	tf.float32	32 bits floating point.
DT_DOUBLE	tf.float64	64 bits floating point.
DT_INT8	tf.int8	8 bits signed integer.
DT_INT16	tf.int16	16 bits signed integer.
DT_INT32	tf.int32	32 bits signed integer.
DT_INT64	tf.int64	64 bits signed integer.
DT_UINT8	tf.uint8	8 bits unsigned integer.
DT_STRING	tf.string	Variable length byte arrays. Each element of a Tensor is a byte array.
DT_BOOL	tf.bool	Boolean.
DT_COMPLEX64	tf.complex64	Complex number made of two 32 bits floating points: real and imaginary parts.
DT_COMPLEX128	tf.complex128	Complex number made of two 64 bits floating points: real and imaginary parts.
DT_QINT8	tf.qint8	8 bits signed integer used in quantized Ops.
DT_QINT32	tf.qint32	32 bits signed integer used in quantized Ops.
DT_QUINT8	tf.quint8	8 bits unsigned integer used in quantized Ops.
'''
# create a placeholder
a = tf.placeholder(tf.float32)
b = a * 2

# since we created a 'hole' in model to pass data, when we initialize the session we are obliged to pass
# an argument with data -- or we will get an error
'''
To pass the data into the model we call the session with an extra argument feed_dict
in which we should pass a dictionary with each placeholder name followed by its respective data
'''
with tf.Session() as sess:
    result = sess.run(b, feed_dict={a:3.5})
    print(result)

print()

# data in TensorFlow is passed in form of multidimensional arrays, we can pass any kind of tensor through placeholders
dictionary={a: [ [ [1,2,3],[4,5,6],[7,8,9],[10,11,12] ] , [ [13,14,15],[16,17,18],[19,20,21],[22,23,24] ] ] }
with tf.Session() as sess:
    result = sess.run(b,feed_dict=dictionary)
    print (result)

print()
##

## OPERATIONS ##
'''
Operations are nodes that represent the mathematical operations over the tensors on a graph.
These operations can be any kind of functions, like add and subtract tensor or maybe an activation function.

tf.constant, tf.matmul, tf.add, tf.nn.sigmoid are some of the operations in TensorFlow.
These are like functions in python but operate directly over tensors and each one does a specific thing.
'''
graph5 = tf.Graph()
with graph5.as_default():
    a = tf.constant([5])
    b = tf.constant([2])
    c = tf.add(a,b)
    d = tf.subtract(a,b)

with tf.Session(graph = graph5) as sess:
    result = sess.run(c)
    print ('c =: %s' % result)
    result = sess.run(d)
    print ('d =: %s' % result)

print()
##
