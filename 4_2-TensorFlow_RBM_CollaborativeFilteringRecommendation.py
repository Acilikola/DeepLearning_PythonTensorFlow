'''
the usage of a Restricted Boltzmann Machine (RBM) in a Collaborative Filtering based recommendation system.
This system is an algorithm that recommends items by trying to find users that are similar to each other
based on their item ratings.

The datasets we are going to use were acquired by GroupLens and contain movies, users and movie ratings
by these users. (\data\moviedataset.zip)
'''
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## extract moviedataset.zip to \data\ and load 
'''
Let's begin by loading in our data with Pandas. The .dat files containing our data are similar to CSV files,
but instead of using the ',' (comma) character to separate entries, it uses '::' (two colons) characters instead.
To let Pandas know that it should separate data points at every '::', we have to specify the sep='::' parameter
when calling the function.

Additionally, we also pass it the header=None parameter due to the fact that our files don't contain any headers
'''
movies_df = pd.read_csv('./data/ml-1m/movies.dat', sep='::', header=None, engine='python')
print(movies_df.head())

print()

ratings_df = pd.read_csv('./data/ml-1m/ratings.dat', sep='::', header=None, engine='python')
print(ratings_df.head())

print()

# rename the columns in these dataframes so we can better convey their data more intuitively
movies_df.columns = ['MovieID', 'Title', 'Genres']
print(movies_df.head())

print()

ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
print(ratings_df.head())

print()
##

##RBM model - Format Data
'''
The Restricted Boltzmann Machine model has two layers of neurons, one of which is what we call a visible input layer
and the other is called a hidden layer. The hidden layer is used to learn features from the information fed
through the input layer. For our model, the input is going to contain X neurons, where X is the amount of movies
in our dataset. Each of these neurons will possess a normalized rating value varying from 0 to 1, where 0 meaning
that a user has not watched that movie and the closer the value is to 1, the more the user likes the movie
that neuron's representing. These normalized values, of course, will be extracted and normalized from the ratings
dataset.

After passing in the input, we train the RBM on it and have the hidden layer learn its features. These features
are what we use to reconstruct the input, which in our case, will predict the ratings for movies that user
hasn't watched, which is exactly what we can use to recommend movies
'''
print(len(movies_df)) #see how many movies we have

print()

# see if movie ID's correspond to the number above
user_rating_df = ratings_df.pivot(index='UserID', columns='MovieID', values='Rating')
print(user_rating_df.head())

print()
# normalize user ratings and put it to a variable 'trX'
norm_user_rating_df = user_rating_df.fillna(0) / 5.0
trX = norm_user_rating_df.values
print(trX[0:5])

print()
##

##RBM model - Set Parameters
'''
We will be arbitrarily setting the number of neurons in the hidden layers to 20.

You can freely set this value to any number you want since each neuron in the hidden layer will end up
learning a feature
'''
hiddenUnits = 20
visibleUnits =  len(user_rating_df.columns)
vb = tf.placeholder("float", [visibleUnits]) #Number of unique movies
hb = tf.placeholder("float", [hiddenUnits]) #Number of features we're going to learn
W = tf.placeholder("float", [visibleUnits, hiddenUnits])

# create visible and hidden layer units and set their activation functions
#Phase 1: Input Processing
v0 = tf.placeholder("float", [None, visibleUnits])
_h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))
#Phase 2: Reconstruction
_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb) 
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

# set RBM training parameters and functions
#Learning rate
alpha = 1.0
#Create the gradients
w_pos_grad = tf.matmul(tf.transpose(v0), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)
#Calculate the Contrastive Divergence to maximize
CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])
#Create methods to update the weights and biases
update_w = W + alpha * CD
update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

# set the error function as Mean Absolute Error
err = v0 - v1
err_sum = tf.reduce_mean(err * err)

# initialize variables as 0s
#Current weight
cur_w = np.zeros([visibleUnits, hiddenUnits], np.float32)
#Current visible unit biases
cur_vb = np.zeros([visibleUnits], np.float32)
#Current hidden unit biases
cur_hb = np.zeros([hiddenUnits], np.float32)
#Previous weight
prv_w = np.zeros([visibleUnits, hiddenUnits], np.float32)
#Previous visible unit biases
prv_vb = np.zeros([visibleUnits], np.float32)
#Previous hidden unit biases
prv_hb = np.zeros([hiddenUnits], np.float32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

'''
we train the RBM with 15 epochs with each epoch using 10 batches with size 100. After training, we print out
a graph with the error by epoch
'''
epochs = 15
batchsize = 100
errors = []
for i in range(epochs):
    for start, end in zip( range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
        batch = trX[start:end]
        cur_w = sess.run(update_w, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_vb = sess.run(update_vb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_nb = sess.run(update_hb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
    errors.append(sess.run(err_sum, feed_dict={v0: trX, W: cur_w, vb: cur_vb, hb: cur_hb}))
    print (errors[-1])
plt.plot(errors)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.show()
##
print()
### RECOMMENDATION
'''
We can now predict movies that an arbitrarily selected user might like. This can be accomplished by
feeding in the user's watched movie preferences into the RBM and then reconstructing the input.
The values that the RBM gives us will attempt to estimate the user's preferences for movies that he hasn't watched
based on the preferences of the users that the RBM was trained on
'''
#select random user
mock_user_id = 215
inputUser = trX[mock_user_id-1].reshape(1, -1)
print(inputUser[0:5])

print()

#Feeding in the user and reconstructing the input
hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
feed = sess.run(hh0, feed_dict={ v0: inputUser, W: prv_w, hb: prv_hb})
rec = sess.run(vv1, feed_dict={ hh0: feed, W: prv_w, vb: prv_vb})
print(rec)

print()

# list the 20 most recommended movies for the user by sorting by their scores given by our model
scored_movies_df_mock = movies_df[movies_df['MovieID'].isin(user_rating_df.columns)]
scored_movies_df_mock = scored_movies_df_mock.assign(RecommendationScore = rec[0])
print(scored_movies_df_mock.sort_values(["RecommendationScore"], ascending=False).head(20))

print()

##recommend movies the user has not watched yet
# find all movies the user has watched before
movies_df_mock = ratings_df[ratings_df['UserID'] == mock_user_id]
print(movies_df_mock.head())

print()
# merge all movies our user has watched with the predicted scores based on historical data
merged_df_mock = scored_movies_df_mock.merge(movies_df_mock, on='MovieID', how='outer')

# sort and check
print(merged_df_mock.sort_values(["RecommendationScore"], ascending=False).head(20))
##
###
