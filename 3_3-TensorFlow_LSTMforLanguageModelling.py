# TENSORFLOW version diff from v1 to v2
# change ==>
## tf.contrib.rnn.* to tf.nn.rnn_cell.*
'''
+Applying Recurrent Neural Networks/LSTM for Language Modeling

In this notebook, we will go over the topic of Language Modelling, and create a Recurrent Neural Network model
based on the Long Short-Term Memory unit to train and benchmark on the Penn Treebank dataset. we go over a TensorFlow
code snippet for creating a model focused on Language Modelling -- a very relevant task that is the cornerstone
of many different linguistic problems such as Speech Recognition, Machine Translation and Image Captioning.
The goal for this notebook is to create a model that can reach low levels of perplexity on our desired dataset.

For Language Modelling problems, perplexity is the way to gauge efficiency.
Perplexity is simply a measure of how well a probabilistic model is able to predict its sample.
A higher-level way to explain this would be saying that low perplexity means a higher degree of trust in
the predictions the model makes. Therefore, the lower perplexity is, the better.

+What exactly is Language Modelling?

Language Modelling, to put it simply, is the task of assigning probabilities to sequences of words.
This means that, given a context of one or a sequence of words in the language the model was trained on,
the model should provide the next most probable words or sequence of words that follows from the
given sequence of words the sentence. Language Modelling is one of the most important tasks in
Natural Language Processing.

+The Penn Treebank dataset

The Penn Treebank, or PTB for short, is a dataset maintained by the University of Pennsylvania.
It is huge -- there are over four million and eight hundred thousand annotated words in it, all corrected by humans.
It is composed of many different sources, from abstracts of Department of Energy papers to texts from the
Library of America. Since it is verifiably correct and of such a huge size, the Penn Treebank is commonly used as
a benchmark dataset for Language Modelling.

The dataset is divided in different kinds of annotations, such as Piece-of-Speech, Syntactic and Semantic skeletons.
For this example, we will simply use a sample of clean, non-annotated words
(with the exception of one tag --<unk> , which is used for rare words such as uncommon proper nouns) for our model.
This means that we just want to predict what the next words would be, not what they mean in context
or their classes on a given sentence

+Word Embeddings

For better processing, in this example, we will make use of word embeddings, which is a way of representing
sentence structures or words as n-dimensional vectors (where n is a reasonably high number, such as 200 or 500)
of real numbers. Basically, we will assign each word a randomly-initialized vector, and input those into the network
to be processed. After a number of iterations, these vectors are expected to assume values that help the network
to correctly predict what it needs to -- in our case, the probable next word in the sentence.
This is shown to be a very effective task in Natural Language Processing, and is a commonplace practice

Word Embedding tends to group up similarly used words reasonably close together in the vectorial space.
For example, "None" is pretty semantically close to "Zero", while a phrase that uses "Italy",
you could probably also fit "Germany" in it, with little damage to the sentence structure.
The vectorial "closeness" for similar words like this is a great indicator of a well-built model.
'''
import time
import numpy as np
import tensorflow as tf

# copy /data/ptb/reader.py to working folder to import
# (https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/reader.py)
import reader

# get example dataset and extract under /data (http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)
#Data directory for our dataset
data_dir = "data/simple-examples/data/"

## define hyperparameters; feel free to change these - will effect performance of model each time they are changed
#Initial weight scale
init_scale = 0.1
#Initial learning rate
learning_rate = 1.0
#Maximum permissible norm for the gradient (For gradient clipping -- another measure against Exploding Gradients)
max_grad_norm = 5
#The number of layers in our model
num_layers = 2
#The total number of recurrence steps, also known as the number of layers when our RNN is "unfolded"
num_steps = 20
#The number of processing units (neurons) in the hidden layers
hidden_size_l1 = 256
hidden_size_l2 = 128
#The maximum number of epochs trained with the initial learning rate
max_epoch_decay_lr = 4
#The total number of epochs in training
max_epoch = 15
#The probability for keeping data in the Dropout Layer (This is an optimization, but is outside our scope for this notebook!)
#At 1, we ignore the Dropout Layer wrapping.
keep_prob = 1
#The decay for the learning rate
decay = 0.5
#The size for each batch of data
batch_size = 60
#The size of our vocabulary
vocab_size = 10000
embeding_vector_size = 200
#Training flag to separate training from testing
is_training = 1
##

'''
LSTM architecture based on the arguments:

Network structure:

- In this network, the number of LSTM cells are 2. To give the model more expressive power,
we can add multiple layers of LSTMs to process the data. The output of the first layer will become the
input of the second and so on.
- The recurrence steps is 20, that is, when our RNN is "Unfolded", the recurrence step is 20.
- the structure is like:
200 input units -> [200x200] Weight -> 200 Hidden units (first layer) -> [200x200] Weight matrix
-> 200 Hidden units (second layer) -> [200] weight Matrix -> 200 unit output

Input layer:

- The network has 200 input units.
- Suppose each word is represented by an embedding vector of dimensionality e=200.
The input layer of each cell will have 200 linear units. These e=200 linear units are connected to each of the
h=200 LSTM units in the hidden layer (assuming there is only one hidden layer, though our case has 2 layers).
- The input shape is [batch_size, num_steps], that is [30x20]. It will turn into [30x20x200] after embedding,
and then 20x[30x200]

Hidden layer:

- Each LSTM has 200 hidden units which is equivalent to the dimensionality of the embedding words and output
'''
## train data
'''
The story starts from data:
- Train data is a list of words, of size 929589, represented by numbers, e.g. [9971, 9972, 9974, 9975,...]
- We read data as mini-batch of size b=30. Assume the size of each sentence is 20 words (num_steps = 20).
Then it will take floor(N/(b x h))+1 - 1548 iterations for the learner to go through all sentences once.
Where N is the size of the list of words, b is batch size, and h is size of each sentence.
So, the number of iterators is 1548
- Each batch data is read from train dataset of size 600, and shape of [30x20]
'''
session = tf.InteractiveSession()

# Reads the data and separates it into training data, validation data and testing data
raw_data = reader.ptb_raw_data(data_dir)
train_data, valid_data, test_data, vocab, word_to_id = raw_data
print(len(train_data))

print()

def id_to_word(id_list):
    line = []
    for w in id_list:
        for word, wid in word_to_id.items():
            if wid == w:
                line.append(word)
    return line            
                
print(id_to_word(train_data[0:100]))

print()

# read one mini-batch and feed our network
itera = reader.ptb_iterator(train_data, batch_size, num_steps)
first_touple = itera.__next__()
x = first_touple[0]
y = first_touple[1]
print(x.shape)
print(x[0:3]) # look at 3 sentences of our input x

print()

_input_data = tf.placeholder(tf.int32, [batch_size, num_steps]) #[30x20]
_targets = tf.placeholder(tf.int32, [batch_size, num_steps]) #[30x20]
feed_dict = {_input_data:x, _targets:y}
print(session.run(_input_data, feed_dict))

print()

# create stacked LSTM with 2 layers
lstm_cell_l1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size_l1, forget_bias=0.0)
lstm_cell_l2 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size_l2, forget_bias=0.0)
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_l1, lstm_cell_l2])

# init states
'''
For each LSTM, there are 2 state matrices, c_state and m_state. c_state and m_state represent "Cell State" and
"Memory State". Each hidden layer, has a vector of size 30, which keeps the states.
so, for 200 hidden units in each LSTM, we have a matrix of size [30x200]
'''
_initial_state = stacked_lstm.zero_state(batch_size, tf.float32)
print(_initial_state)
print()
print(session.run(_initial_state, feed_dict)) # look at the states, though all zero for now

print()
##
## EMBEDDINGS
'''
We have to convert the words in our dataset to vectors of numbers. The traditional approach is to use
one-hot encoding method that is usually used for converting categorical values to numerical values.
However, One-hot encoded vectors are high-dimensional, sparse and in a big dataset, computationally inefficient.
So, we use word2vec approach. It is, in fact, a layer in our LSTM network, where the word IDs will be represented
as a dense representation before feeding to the LSTM.

The embedded vectors also get updated during the training process of the deep neural network.
We create the embeddings for our input data. embedding_vocab is matrix of [10000x200] for all 10000 unique words
'''
embedding_vocab = tf.get_variable("embedding_vocab", [vocab_size, embeding_vector_size])  #[10000x200]
session.run(tf.global_variables_initializer())
print(session.run(embedding_vocab)) # initialize embedding_words with random values

print()

'''
embedding_lookup() finds the embedded values for our batch of 30x20 words.
It goes to each row of input_data, and for each word in the row/sentence, finds the correspond vector in embedding_dic

It creates a [30x20x200] tensor, so, the first element of inputs (the first sentence), is a matrix of 20x200,
which each row of it, is vector representing a word in the sentence
'''
# Define where to get the data for our embeddings from
inputs = tf.nn.embedding_lookup(embedding_vocab, _input_data)  #shape=(30, 20, 200) 
print(inputs)
print(session.run(inputs[0], feed_dict))
##
## Construct RNN
'''
tf.nn.dynamic_rnn() creates a recurrent neural network using stacked_lstm.
The input should be a Tensor of shape: [batch_size, max_time, embedding_vector_size],
in our case it would be (30, 20, 200)

This method, returns a pair (outputs, new_state) where:
- outputs: is a length T list of outputs (one for each input), or a nested tuple of such elements.
- new_state: is the final state.
'''
outputs, new_state =  tf.nn.dynamic_rnn(stacked_lstm, inputs, initial_state=_initial_state)
print(outputs)

# look at outputs
'''
output of the stackedLSTM comes from 200 hidden_layer, and in each time step(=20), one of them get activated.
we use the linear activation to map the 200 hidden layer to a [?x10 matrix]
'''
session.run(tf.global_variables_initializer())
print(session.run(outputs[0], feed_dict))

print()

# flatten outputs
'''
we need to flatten the outputs to be able to connect it softmax layer. Lets reshape the output tensor from
[30 x 20 x 200] to [600 x 200].

Notice: Imagine our output is 3-d tensor as following (of course each sen_x_word_y is a an embedded vector by itself):

sentence 1: [[sen1word1], [sen1word2], [sen1word3], ..., [sen1word20]]
sentence 2: [[sen2word1], [sen2word2], [sen2word3], ..., [sen2word20]]
sentence 3: [[sen3word1], [sen3word2], [sen3word3], ..., [sen3word20]]
...
sentence 30: [[sen30word1], [sen30word2], [sen30word3], ..., [sen30word20]]

Now, the flatten would convert this 3-dim tensor to:
[ [sen1word1], [sen1word2], [sen1word3], ..., [sen1word20],[sen2word1], [sen2word2], [sen2word3],
..., [sen2word20], ..., [sen30word20] ]
'''
output = tf.reshape(outputs, [-1, hidden_size_l2])
print(output)

print()

# softmax + logistic unit
'''
we create a logistic unit to return the probability of the output word in our vocabulary with 1000 words.
Softmax = [600 x 200]*[200 x 1000] + [1 x 1000] ==> [600 x 1000]
'''
softmax_w = tf.get_variable("softmax_w", [hidden_size_l2, vocab_size]) #[200x1000]
softmax_b = tf.get_variable("softmax_b", [vocab_size]) #[1x1000]
logits = tf.matmul(output, softmax_w) + softmax_b
prob = tf.nn.softmax(logits)

# look at probability of observing words for t=0 to t=20
session.run(tf.global_variables_initializer())
output_words_prob = session.run(prob, feed_dict)
print("shape of the output: ", output_words_prob.shape)
print("The probability of observing words in t=0 to t=20", output_words_prob[0:20])

print()
##
## PREDICTION
'''
What is the word correspond to the probability output? Lets use the maximum probability
'''
print(np.argmax(output_words_prob[0:20], axis=1)) #predicted

print()

print(y[0]) #ground truth for the first word of first sentence

print()

#ground truth for the first word of first sentence
#get it from target tensor, if you want to find the embedding vector
targ = session.run(_targets, feed_dict)
print(targ[0])

print()
##
## OBJECTIVE FUNCTION
'''
Now we have to define our objective function, to calculate the similarity of predicted values to ground truth,
and then, penalize the model with the error. Our objective is to minimize loss function, that is, to minimize
the average negative log probability of the target words:

loss = -1/N * <SUM>{ln(p_target_i)}

This function is already implemented and available in TensorFlow through sequence_loss_by_example.
It calculates the weighted cross-entropy loss for logits and the target sequence.
The arguments of this function are:
- logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
- targets: List of 1D batch-sized int32 Tensors of the same length as logits.
- weights: List of 1D batch-sized float-Tensors of the same length as logits.
'''
loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(_targets, [-1])],[tf.ones([batch_size * num_steps])])
# loss is a 1D batch-sized float Tensor [600x1]: The log-perplexity for each sequence

print(session.run(loss, feed_dict)[:10]) #look at first 10 values of loss

print()

# define loss as avg of losses
cost = tf.reduce_sum(loss) / batch_size
session.run(tf.global_variables_initializer())
print(session.run(cost, feed_dict))

print()
##
### TRAINING
'''
- Define the optimizer.
- Extract variables that are trainable.
- Calculate the gradients based on the loss function.
- Apply the optimizer to the variables/gradients tuple.
'''
## define optimizer
'''
GradientDescentOptimizer constructs a new gradient descent optimizer. Later, we use constructed optimizer
to compute gradients for a loss and apply gradients to variables
'''
# Create a variable for the learning rate
lr = tf.Variable(0.0, trainable=False)
# Create the gradient descent optimizer with our learning rate
optimizer = tf.train.GradientDescentOptimizer(lr)
##
## trainable variables
'''
Defining a variable, if you passed trainable=True, the variable constructor automatically adds new variables
to the graph collection GraphKeys.TRAINABLE_VARIABLES. Now, using tf.trainable_variables() you can get all
variables created with trainable=True
'''
# Get all TensorFlow variables marked as "trainable" (i.e. all of them except _lr, which we just created)
tvars = tf.trainable_variables()
print(tvars)

print()

print([v.name for v in tvars]) #find the name and scope of all variables

print()
##
## calculate gradients based on loss func
'''
The gradient of a function is the slope of its derivative (line), or in other words, the rate of change of a function.
It's a vector (a direction to move) that points in the direction of greatest increase of the function,
and calculated by the derivative operation

The tf.gradients() function allows you to compute the symbolic gradient of one tensor with respect to
one or more other tensors—including variables. tf.gradients(func, xs) constructs symbolic partial derivatives of
sum of func w.r.t. x in xs
'''
grad_t_list = tf.gradients(cost, tvars)
print(grad_t_list) #look at gradients w.r.t all variables

print()
'''
now, we have a list of tensors, t-list. We can use it to find clipped tensors.
clip_by_global_norm clips values of multiple tensors by the ratio of the sum of their norms.

clip_by_global_norm get t-list as input and returns 2 things:
- a list of clipped tensors, so called list_clipped
- the global norm (global_norm) of all tensors in t_list
'''
# Define the gradient clipping threshold
grads, _ = tf.clip_by_global_norm(grad_t_list, max_grad_norm)
print(grads)

print()

print(session.run(grads, feed_dict))

print()
##
## apply the optimizer to the variables / gradients tuple
# Create the training TensorFlow Operation through our optimizer
train_op = optimizer.apply_gradients(zip(grads, tvars))

session.run(tf.global_variables_initializer())
print(session.run(train_op, feed_dict))

print()
##
###

### LSTM
'''
We learned how the model is build step by step. Now, let's then create a Class that represents our model.
This class needs a few things:

- We have to create the model in accordance with our defined hyperparameters
- We have to create the placeholders for our input data and expected outputs (the real data)
- We have to create the LSTM cell structure and connect them with our RNN structure
- We have to create the word embeddings and point them to the input data
- We have to create the input structure for our RNN
- We have to instantiate our RNN model and retrieve the variable in which we should expect our outputs to appear
- We need to create a logistic structure to return the probability of our words
- We need to create the loss and cost functions for our optimizer to work, and then create the optimizer
- And finally, we need to create a training operation that can be run to actually train our model
'''
print(hidden_size_l1)

print()

class PTBModel(object):

    def __init__(self, action_type):
        ######################################
        # Setting parameters for ease of use #
        ######################################
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.hidden_size_l1 = hidden_size_l1
        self.hidden_size_l2 = hidden_size_l2
        self.vocab_size = vocab_size
        self.embeding_vector_size = embeding_vector_size
        ###############################################################################
        # Creating placeholders for our input data and expected outputs (target data) #
        ###############################################################################
        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps]) #[30#20]
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps]) #[30#20]

        ##########################################################################
        # Creating the LSTM cell structure and connect it with the RNN structure #
        ##########################################################################
        # Create the LSTM unit. 
        # This creates only the structure for the LSTM and has to be associated with a RNN unit still.
        # The argument n_hidden(size=200) of BasicLSTMCell is size of hidden layer, that is, the number of hidden units of the LSTM (inside A).
        # Size is the same as the size of our hidden layer, and no bias is added to the Forget Gate. 
        # LSTM cell processes one word at a time and computes probabilities of the possible continuations of the sentence.
        lstm_cell_l1 = tf.contrib.rnn.BasicLSTMCell(self.hidden_size_l1, forget_bias=0.0)
        lstm_cell_l2 = tf.contrib.rnn.BasicLSTMCell(self.hidden_size_l2, forget_bias=0.0)
        
        # Unless you changed keep_prob, this won't actually execute -- this is a dropout wrapper for our LSTM unit
        # This is an optimization of the LSTM output, but is not needed at all
        if action_type == "is_training" and keep_prob < 1:
            lstm_cell_l1 = tf.contrib.rnn.DropoutWrapper(lstm_cell_l1, output_keep_prob=keep_prob)
            lstm_cell_l2 = tf.contrib.rnn.DropoutWrapper(lstm_cell_l2, output_keep_prob=keep_prob)
        
        # By taking in the LSTM cells as parameters, the MultiRNNCell function junctions the LSTM units to the RNN units.
        # RNN cell composed sequentially of multiple simple cells.
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell_l1, lstm_cell_l2])

        # Define the initial state, i.e., the model state for the very first data point
        # It initialize the state of the LSTM memory. The memory state of the network is initialized with a vector of zeros and gets updated after reading each word.
        self._initial_state = stacked_lstm.zero_state(batch_size, tf.float32)

        ####################################################################
        # Creating the word embeddings and pointing them to the input data #
        ####################################################################
        with tf.device("/cpu:0"):
            # Create the embeddings for our input data. Size is hidden size.
            embedding = tf.get_variable("embedding", [vocab_size, self.embeding_vector_size])  #[10000x200]
            # Define where to get the data for our embeddings from
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        # Unless you changed keep_prob, this won't actually execute -- this is a dropout addition for our inputs
        # This is an optimization of the input processing and is not needed at all
        if action_type == "is_training" and keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob)

        ############################################
        # Creating the input structure for our RNN #
        ############################################
        # Input structure is 20x[30x200]
        # Considering each word is represended by a 200 dimentional vector, and we have 30 batchs, we create 30 word-vectors of size [30xx2000]
        # inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, num_steps, inputs)]
        # The input structure is fed from the embeddings, which are filled in by the input data
        # Feeding a batch of b sentences to a RNN:
        # In step 1,  first word of each of the b sentences (in a batch) is input in parallel.  
        # In step 2,  second word of each of the b sentences is input in parallel. 
        # The parallelism is only for efficiency.  
        # Each sentence in a batch is handled in parallel, but the network sees one word of a sentence at a time and does the computations accordingly. 
        # All the computations involving the words of all sentences in a batch at a given time step are done in parallel. 

        ####################################################################################################
        # Instantiating our RNN model and retrieving the structure for returning the outputs and the state #
        ####################################################################################################
        
        outputs, state = tf.nn.dynamic_rnn(stacked_lstm, inputs, initial_state=self._initial_state)
        #########################################################################
        # Creating a logistic unit to return the probability of the output word #
        #########################################################################
        output = tf.reshape(outputs, [-1, self.hidden_size_l2])
        softmax_w = tf.get_variable("softmax_w", [self.hidden_size_l2, vocab_size]) #[200x1000]
        softmax_b = tf.get_variable("softmax_b", [vocab_size]) #[1x1000]
        logits = tf.matmul(output, softmax_w) + softmax_b
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])
        prob = tf.nn.softmax(logits)
        out_words = tf.argmax(prob, axis=2)
        self._output_words = out_words
        #########################################################################
        # Defining the loss and cost functions for the model's learning to work #
        #########################################################################
            

        # Use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            self.targets,
            tf.ones([batch_size, num_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)
    
#         loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(self._targets, [-1])],
#                                                       [tf.ones([batch_size * num_steps])])
        self._cost = tf.reduce_sum(loss)

        # Store the final state
        self._final_state = state

        #Everything after this point is relevant only for training
        if action_type != "is_training":
            return

        #################################################
        # Creating the Training Operation for our Model #
        #################################################
        # Create a variable for the learning rate
        self._lr = tf.Variable(0.0, trainable=False)
        # Get all TensorFlow variables marked as "trainable" (i.e. all of them except _lr, which we just created)
        tvars = tf.trainable_variables()
        # Define the gradient clipping threshold
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), max_grad_norm)
        # Create the gradient descent optimizer with our learning rate
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        # Create the training TensorFlow Operation through our optimizer
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    # Helper functions for our LSTM RNN class

    # Assign the learning rate for this model
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    # Returns the input data for this model at a point in time
    @property
    def input_data(self):
        return self._input_data


    
    # Returns the targets for this model at a point in time
    @property
    def targets(self):
        return self._targets

    # Returns the initial state for this model
    @property
    def initial_state(self):
        return self._initial_state

    # Returns the defined Cost
    @property
    def cost(self):
        return self._cost

    # Returns the final state for this model
    @property
    def final_state(self):
        return self._final_state
    
    # Returns the final output words for this model
    @property
    def final_output_words(self):
        return self._output_words
    
    # Returns the current learning rate for this model
    @property
    def lr(self):
        return self._lr

    # Returns the training operation defined for this model
    @property
    def train_op(self):
        return self._train_op

'''
With that, the actual structure of our Recurrent Neural Network with Long Short-Term Memory is finished.
What remains for us to do is to actually create the methods to run through time --
that is, the run_epoch method to be run at each epoch and a main script which ties all of this together.

What our run_epoch method should do is take our input data and feed it to the relevant operations.
This will return at the very least the current result for the cost function
'''
##########################################################################################################################
# run_one_epoch takes as parameters the current session, the model instance, the data to be fed, and the operation to be run #
##########################################################################################################################
def run_one_epoch(session, m, data, eval_op, verbose=False):

    #Define the epoch size based on the length of the data, batch size and the number of steps
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0

    state = session.run(m.initial_state)
    
    #For each step and data point
    for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size, m.num_steps)):
        
        #Evaluate and return cost, state by running cost, final_state and the function passed as parameter
        cost, state, out_words, _ = session.run([m.cost, m.final_state, m.final_output_words, eval_op],
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_state: state})

        #Add returned cost to costs (which keeps track of the total costs for this epoch)
        costs += cost
        
        #Add number of steps to iteration counter
        iters += m.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("Itr %d of %d, perplexity: %.3f speed: %.0f wps" % (step , epoch_size, np.exp(costs / iters), iters * m.batch_size / (time.time() - start_time)))

    # Returns the Perplexity rating for us to keep track of how the model is evolving
    return np.exp(costs / iters)

'''
Now, we create the main method to tie everything together. The code here reads the data from the directory,
using the reader helper module, and then trains and evaluates the model on both a testing and
a validating subset of data
'''
# Reads the data and separates it into training data, validation data and testing data
raw_data = reader.ptb_raw_data(data_dir)
train_data, valid_data, test_data, _, _ = raw_data

# Initializes the Execution Graph and the Session
with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    
    # Instantiates the model for training
    # tf.variable_scope add a prefix to the variables created with tf.get_variable
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        m = PTBModel("is_training")
        
    # Reuses the trained parameters for the validation and testing models
    # They are different instances but use the same variables for weights and biases, they just don't change when data is input
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        mvalid = PTBModel("is_validating")
        mtest = PTBModel("is_testing")

    #Initialize all variables
    tf.global_variables_initializer().run()

    for i in range(max_epoch):
        # Define the decay for this epoch
        lr_decay = decay ** max(i - max_epoch_decay_lr, 0.0)
        
        # Set the decayed learning rate as the learning rate for this epoch
        m.assign_lr(session, learning_rate * lr_decay)

        print("Epoch %d : Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        
        # Run the loop for this epoch in the training model
        train_perplexity = run_one_epoch(session, m, train_data, m.train_op, verbose=True)
        print("Epoch %d : Train Perplexity: %.3f" % (i + 1, train_perplexity))
        
        # Run the loop for this epoch in the validation model
        valid_perplexity = run_one_epoch(session, mvalid, valid_data, tf.no_op())
        print("Epoch %d : Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
    
    # Run the loop in the testing model to see how effective was our training
    test_perplexity = run_one_epoch(session, mtest, test_data, tf.no_op())
    
    print("Test Perplexity: %.3f" % test_perplexity)

###
'''
As you can see, the model's perplexity rating drops very quickly after a few iterations.
As was elaborated before, lower Perplexity means that the model is more certain about its prediction.
As such, we can be sure that this model is performing well!
'''
