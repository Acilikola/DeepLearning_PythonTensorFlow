# TENSORFLOW version diff from v1 to v2
# change ==>
## tf.contrib.rnn.* to tf.nn.rnn_cell.*
'''
+Applying Recurrent Neural Networks/LSTM for Character Modeling

This code implements a Recurrent Neural Network with LSTM/RNN units for training/sampling from
character-level language models. In other words, the model takes a text file as input and trains the
RNN network that learns to predict the next character in a sequence.

The RNN can then be used to generate text character by character that will look like the original training data.

This code is based on this blog (http://karpathy.github.io/2015/05/21/rnn-effectiveness/), and the code is an
step-by-step implimentation of the character-level implimentation(https://github.com/crazydonkey200/tensorflow-char-rnn)
'''
import time
import numpy as np
import tensorflow as tf
import codecs
import os
import collections
from six.moves import cPickle

# a class to help read data from input file
class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))

        # When the data (tensor) is too small, let's give them a better error message
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)


    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

# parameters
'''
lets assume the input is this sentence: 'here is an example'. Then:

txt_length = 18
seq_length = 3
batch_size = 2
number_of_batchs = 18/3*2 = 3
batch = array (['h','e','r'],['e',' ','i'])
sample Seq = 'her'
'''
seq_length = 50 # RNN sequence length
batch_size = 60  # minibatch size, i.e. size of data in each epoch
num_epochs = 125 # you should change it to 50 if you want to see a relatively good results
learning_rate = 0.002
decay_rate = 0.97
rnn_size = 128 # size of RNN hidden state (output dimension)
num_layers = 2 #number of layers in the RNN

# load input file from 'input.txt' under work directory
with open('input.txt', 'r') as f:
    read_data = f.read()
    print (read_data[0:100])
f.closed

print()
# read data at batches using the class above 'TextLoader'
'''
will convert the characters to numbers, and represent each sequence as a vector in batches
'''
data_loader = TextLoader('', batch_size, seq_length)
vocab_size = data_loader.vocab_size
print ("vocabulary size:" ,data_loader.vocab_size)
print ("Characters:" ,data_loader.chars)
print ("vocab number of 'F':",data_loader.vocab['F'])
print ("Character sequences (first batch):", data_loader.x_batches[0])

print()

# define input and output
x,y = data_loader.next_batch()
print(x)
print()
print(x.shape)
print()
print(y)
print()
print(y.shape)

print()

# LSTM architecture
'''
Each LSTM cell has 5 parts:
 - Input
 - prv_state
 - prv_output
 - new_state
 - new_output
 
Each LSTM cell has an input layer, which its size is 128 units in our case.The input vector's dimension also is 128,
which is the dimensionality of embedding vector, so called, dimension size of W2V/embedding, for each character/word.

Each LSTM cell has a hidden layer, where there are some hidden units. The argument n_hidden=128 of BasicLSTMCell is
the number of hidden units of the LSTM (inside A). It keeps the size of the output and state vector.
It is also known as, rnn_size, num_units, num_hidden_units, and LSTM size

An LSTM keeps two pieces of information as it propagates through time:
- hidden state vector: Each LSTM cell accept a vector, called hidden state vector, of size n_hidden=128, and
its value is returned to the LSTM cell in the next step. The hidden state vector; which is the memory of the LSTM,
accumulates using its (forget, input, and output) gates through time. "num_units" is equivalant to
"size of RNN hidden state". number of hidden units is the dimensianality of
the output (= dimesianality of the state) of the LSTM cell.
- previous time-step output: For each LSTM cell that we initialize, we need to supply a value (128 in this case)
for the hidden dimension, or as some people like to call it, the number of units in the LSTM cell.

num_layers = 2
- number of layers in the RNN, is defined by num_layers
- An input of MultiRNNCell is cells which is list of RNNCells that will be composed in this order.
'''
cell = tf.contrib.rnn.BasicRNNCell(rnn_size)
# a two layer cell
stacked_cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
# hidden state size
print(stacked_cell.output_size)
# state variable keeps output and new_state
print(stacked_cell.state_size)

print()
# define input and target
input_data = tf.placeholder(tf.int32, [batch_size, seq_length]) #60x50
print(input_data)
targets = tf.placeholder(tf.int32, [batch_size, seq_length]) #60x50
print(targets)

print()

initial_state = stacked_cell.zero_state(batch_size, tf.float32) #60x128

session = tf.Session()
feed_dict={input_data:x, targets:y}
print(session.run(input_data, feed_dict))

print()

## Embedding
'''
we build a 128-dim vector for each character. As we have 60 batches, and 50 character in each sequence,
it will generate a [60,50,128] matrix

The function tf.get_variable() is used to share a variable and to initialize it in one place.
tf.get_variable() is used to get or create a variable instead of a direct call to tf.Variable
'''
with tf.variable_scope('rnnlm', reuse=False):
    softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size]) #128x65
    softmax_b = tf.get_variable("softmax_b", [vocab_size]) # 1x65)
    #with tf.device("/cpu:0"):
        
    # embedding variable is initialized randomely
    embedding = tf.get_variable("embedding", [vocab_size, rnn_size])  #65x128

    # embedding_lookup goes to each row of input_data, and for each character in the row, finds the correspond vector in embedding
    # it creates a 60*50*[1*128] matrix
    # so, the first elemnt of em, is a matrix of 50x128, which each row of it is vector representing that character
    em = tf.nn.embedding_lookup(embedding, input_data) # em is 60x50x[1*128]
    # split: Splits a tensor into sub tensors.
    # syntax:  tf.split(split_dim, num_split, value, name='split')
    # it will split the 60x50x[1x128] matrix into 50 matrix of 60x[1*128]
    inputs = tf.split(em, seq_length, 1)
    # It will convert the list to 50 matrix of [60x128]
    inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

# take a look at embedding, em and inputs variables
session.run(tf.global_variables_initializer())


print(embedding.shape)
print()
print(session.run(embedding))

print()

em = tf.nn.embedding_lookup(embedding, input_data)
emp = session.run(em,feed_dict={input_data:x})
print(emp.shape)
print()
print(emp[0]) #first element of em, is a matrix of 50x128, with each row of it is vector representing that character

print()

# consider each sequence as a sentence of length 50 characters, then, the first item in inputs is a [60x128] vector
# which represents the first characters of 60 sentences
inputs = tf.split(em, seq_length, 1)
inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
print(inputs[0:5])

print()
##
## FEEDING BATCH to RNN
'''
Feeding a batch of 50 sequence to a RNN:

The feeding process for inputs is as following:
Step 1: first character of each of the 50 sentences (in a batch) is entered in parallel.
Step 2: second character of each of the 50 sentences is input in parallel.
Step n: nth character of each of the 50 sentences is input in parallel.

The parallelism is only for efficiency. Each character in a batch is handled in parallel, but the network sees
one character of a sequence at a time and does the computations accordingly. All the computations involving
the characters of all sequences in a batch at a given time step are done in parallel
'''
session.run(inputs[0],feed_dict={input_data:x})

'''
Feeding the RNN with one batch, we can check the new output and new state of network
'''
#outputs is 50x[60*128]
outputs, new_state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, initial_state, stacked_cell, loop_function=None, scope='rnnlm')
print(new_state)
print()
print(outputs[0:5])

print()

#check the output of network after feeding it with first batch
first_output = outputs[0]
session.run(tf.global_variables_initializer())
print(session.run(first_output,feed_dict={input_data:x}))

print()

# reshape output & define logits and probs to use with Softmax fully connected layer
'''
As it was explained, outputs variable is a 50x[60x128] tensor. We need to reshape it back to [60x50x128]
to be able to calculate the probablity of the next character using the softmax. The softmax_w shape is
[rnn_size, vocab_size],which is [128x65] in our case. Therefore, we have a fully connected layer on top of LSTM cells,
which help us to decode the next character. We can use the softmax(output * softmax_w + softmax_b) for this purpose.
The shape of the matrixis would be:
    softmax([60x50x128]x[128x65]+[1x65]) = [60x50x65]
'''
output = tf.reshape(tf.concat( outputs,1), [-1, rnn_size])
print(output)

print()

logits = tf.matmul(output, softmax_w) + softmax_b
print(logits)

print()

probs = tf.nn.softmax(logits)
print(probs)

print()

# probability of the next character in all batches
session.run(tf.global_variables_initializer())
print(session.run(probs,feed_dict={input_data:x}))

print()

# calculate cost of training with loss function and feed network to learn it
grad_clip =5.
tvars = tf.trainable_variables()
print(tvars)

print()
##

## ALL TOGETHER
'''
put all of parts together in a class, and train the model:
'''
class LSTMModel():
    def __init__(self,sample=False):
        rnn_size = 128 # size of RNN hidden state vector
        batch_size = 60 # minibatch size, i.e. size of dataset in each epoch
        seq_length = 50 # RNN sequence length
        num_layers = 2 # number of layers in the RNN
        vocab_size = 65
        grad_clip = 5.
        if sample:
            print(">> sample mode:")
            batch_size = 1
            seq_length = 1
        # The core of the model consists of an LSTM cell that processes one char at a time and computes probabilities of the possible continuations of the char. 
        basic_cell = tf.contrib.rnn.BasicRNNCell(rnn_size)
        # model.cell.state_size is (128, 128)
        self.stacked_cell = tf.contrib.rnn.MultiRNNCell([basic_cell] * num_layers)

        self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length], name="input_data")
        self.targets = tf.placeholder(tf.int32, [batch_size, seq_length], name="targets")
        # Initial state of the LSTM memory.
        # The memory state of the network is initialized with a vector of zeros and gets updated after reading each char. 
        self.initial_state = stacked_cell.zero_state(batch_size, tf.float32) #why batch_size

        with tf.variable_scope('rnnlm_class1'):
            softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size]) #128x65
            softmax_b = tf.get_variable("softmax_b", [vocab_size]) # 1x65
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [vocab_size, rnn_size])  #65x128
                inputs = tf.split(tf.nn.embedding_lookup(embedding, self.input_data), seq_length, 1)
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
                #inputs = tf.split(em, seq_length, 1)
                
                


        # The value of state is updated after processing each batch of chars.
        outputs, last_state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, self.initial_state, self.stacked_cell, loop_function=None, scope='rnnlm_class1')
        output = tf.reshape(tf.concat(outputs,1), [-1, rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([batch_size * seq_length])],
                vocab_size)
        self.cost = tf.reduce_sum(loss) / batch_size / seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
    
    
    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        state = sess.run(self.stacked_cell.zero_state(1, tf.float32))
        #print state
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret

# create a LSTM model
with tf.variable_scope("rnn"):
    model = LSTMModel()

# train using LSTMModel class above; we can train our model through feeding batches
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(num_epochs): # num_epochs is 5 for test, but should be higher
        sess.run(tf.assign(model.lr, learning_rate * (decay_rate ** e)))
        data_loader.reset_batch_pointer()
        state = sess.run(model.initial_state) # (2x[60x128])
        for b in range(data_loader.num_batches): #for each batch
            start = time.time()
            x, y = data_loader.next_batch()
            feed = {model.input_data: x, model.targets: y, model.initial_state:state}
            train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
            end = time.time()
        print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                .format(e * data_loader.num_batches + b, num_epochs * data_loader.num_batches, e, train_loss, end - start))
        with tf.variable_scope("rnn", reuse=True):
            sample_model = LSTMModel(sample=True)
            print (sample_model.sample(sess, data_loader.chars , data_loader.vocab, num=50, prime='The ', sampling_type=1))
            print ('----------------------------------')

##
