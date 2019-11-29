# TENSORFLOW version diff from v1 to v2
# change ==>
## tf.contrib.rnn.* to tf.nn.rnn_cell.*
'''
Recurrent Neural Networks are Deep Learning models with simple structures and a feedback mechanism built-in,
or in different words, the output of a layer is added to the next input and fed back to the same layer.

The Recurrent Neural Network is a specialized type of Neural Network that solves the issue of
maintaining context for Sequential data -- such as Weather data, Stocks, Genes, etc.
At each iterative step, the processing unit takes in an input and the current state of the network,
and produces an output and a new state that is re-fed into the network

The Long Short-Term Memory (LSTM), was an abstraction of how computer memory works.
It is "bundled" with whatever processing unit is implemented in the Recurrent Network, although outside of its flow,
and is responsible for keeping, reading, and outputting information for the model.

The way it works is simple: you have a linear unit, which is the information cell itself,
surrounded by three logistic gates responsible for maintaining the data.
One gate is for inputting data into the information cell,
one is for outputting data from the input cell,
and the last one is to keep or forget data depending on the needs of the network.

Thanks to that, it not only solves the problem of keeping states,
because the network can choose to forget data whenever information is not needed,
it also solves the gradient problems, since the Logistic Gates have a very nice derivative.

he Long Short-Term Memory is composed of a linear unit surrounded by three logistic gates.
The name for these gates vary from place to place, but the most usual names for them are:

    - "Input" or "Write" Gate, which handles the writing of data into the information cell
    - "Output" or "Read" Gate, which handles the sending of data back onto the Recurrent Network
    - "Keep" or "Forget" Gate, which handles the maintaining and modification of data stored in the information cell

These gates are analog and multiplicative, and as such, can modify the data based on the signal they are sent.
'''
import numpy as np
import tensorflow as tf

## SIMPLE LSTM ##
sess = tf.Session()

# create network with only one LSTM cell
'''
We have to pass 2 elements to LSTM, the prv_output and prv_state, so called, h and c.
Therefore, we initialize a state vector, state.
Here, state is a tuple with 2 elements, each one is of size [1 x 4], one for passing prv_output to next time step,
and another for passing the prv_state to next time stamp
'''
LSTM_CELL_SIZE = 4  # output size (dimension), which is same as hidden size in the cell

#lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_CELL_SIZE, state_is_tuple=True)
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_CELL_SIZE, state_is_tuple=True)
state = (tf.zeros([1,LSTM_CELL_SIZE]),)*2
print(state)

print()

# define a simple input with batch_size = 1 & seq_len = 6; and pass it
sample_input = tf.constant([[3,2,2,2,2,2]],dtype=tf.float32)
print (sess.run(sample_input))

with tf.variable_scope("LSTM_sample1"):
    output, state_new = lstm_cell(sample_input, state)
sess.run(tf.global_variables_initializer())
print (sess.run(state_new))
'''
states has 2 parts, the new state c, and also the output h
'''
print()

print(sess.run(output))

print()
sess.close()
##
## STACKED LSTM ##
'''
What about if we want to have a RNN with stacked LSTM? For example, a 2-layer LSTM.
In this case, the output of the first layer will become the input of the second
'''
sess = tf.Session()
input_dim = 6
cells = []

# create first layer LSTM cell
LSTM_CELL_SIZE_1 = 4 #4 hidden nodes
cell1 = tf.nn.rnn_cell.LSTMCell(LSTM_CELL_SIZE_1)
cells.append(cell1)

# create second layer LSTM cell
LSTM_CELL_SIZE_2 = 5 #5 hidden nodes
cell2 = tf.nn.rnn_cell.LSTMCell(LSTM_CELL_SIZE_2)
cells.append(cell2)

# create multi layer LSTM
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(cells)

# create RNN from stacked_lstm
# Batch size x time steps x features.
data = tf.placeholder(tf.float32, [None, None, input_dim])
output, state = tf.nn.dynamic_rnn(stacked_lstm, data, dtype=tf.float32)

# create input
sample_input = [[[1,2,3,4,3,2], [1,2,1,1,1,2],[1,2,2,2,2,2]],[[1,2,3,4,3,2],[3,2,2,1,1,2],[0,0,0,0,3,2]]]
print(sample_input)

print()

# send input to network and check output
print(output)
print()

sess.run(tf.global_variables_initializer())
print(sess.run(output, feed_dict={data: sample_input}))
'''
As you see, the output is of shape (2, 3, 5), which corresponds to our 2 batches, 3 elements in our sequence,
and the dimensionality of the output which is 5
'''
sess.close()
##
