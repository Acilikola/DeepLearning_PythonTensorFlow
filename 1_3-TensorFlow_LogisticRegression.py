'''
Logistic Regression is a variation of Linear Regression, useful when the observed dependent variable,
y, is categorical. It produces a formula that predicts the probability of the class label as a function
of the independent variables.

Despite the name logistic regression, it is actually a probabilistic classification model.
Logistic regression fits a special s-shaped curve by taking the linear regression and
transforming the numeric estimate into a probability

Probability of a class = p = e^y / 1 + e^y = 1 / 1 + e^-y

Logistic Regression equation ==> y = sigmoid(WX + b)
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tensorflow as tf

'''
Iris Dataset:

This dataset was introduced by British Statistician and Biologist Ronald Fisher,
it consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor).
In total it has 150 records under five attributes - petal length, petal width, sepal length, sepal width and species

Attributes Independent Variable
+petal length
+petal width
+sepal length
+sepal width

Dependent Variable
+Species
++Iris setosa
++Iris virginica
++Iris versicolor
'''
# load iris dataset
iris = load_iris()
iris_X, iris_y = iris.data[:-1,:], iris.target[:-1]
iris_y= pd.get_dummies(iris_y).values
trainX, testX, trainY, testY = train_test_split(iris_X, iris_y, test_size=0.33, random_state=42)

## define x & y; use placeholders
'''
define x and y. These placeholders will hold our iris data (both the features and label matrices),
and help pass them along to different parts of the algorithm.
You can consider placeholders as empty shells into which we insert our data.
We also need to give them shapes which correspond to the shape of our data.
Later, we will insert data into these placeholders by “feeding” the placeholders the data
via a “feed_dict” (Feed Dictionary

Why use Placeholders?

1) This feature of TensorFlow allows us to create an algorithm which accepts data and knows something about
the shape of the data without knowing the amount of data going in.

2) When we insert “batches” of data in training, we can easily adjust how many examples we train on
in a single step without changing the entire algorithm
'''
# numFeatures is the number of features in our input data.
# In the iris dataset, this number is '4'.
numFeatures = trainX.shape[1]

# numLabels is the number of classes our data points can be in.
# In the iris dataset, this number is '3'.
numLabels = trainY.shape[1]

# Placeholders
# 'None' means TensorFlow shouldn't expect a fixed number in that dimension
X = tf.placeholder(tf.float32, [None, numFeatures]) # Iris has 4 features, so X is a tensor to hold our data.
yGold = tf.placeholder(tf.float32, [None, numLabels]) # This will be our correct answers matrix for 3 classes.

##

## set model weights and bias
'''
Much like Linear Regression, we need a shared variable weight matrix for Logistic Regression.
We initialize both W and b as tensors full of zeros. Since we are going to learn W and b,
their initial value does not matter too much. These variables are the objects which
define the structure of our regression model, and we can save them after they have been trained
so we can reuse them later.

We define two TensorFlow variables as our parameters. These variables will hold the weights and biases
of our logistic regression and they will be continually updated during training.

Notice that W has a shape of [4, 3] because we want to multiply the 4-dimensional input vectors
by it to produce 3-dimensional vectors of evidence for the difference classes.

b has a shape of [3] so we can add it to the output.

Moreover, unlike our placeholders above which are essentially empty shells waiting to be fed data,
TensorFlow variables need to be initialized with values, e.g. with zeros
'''
W = tf.Variable(tf.zeros([4, 3]))  # 4-dimensional input and  3 classes
b = tf.Variable(tf.zeros([3])) # 3-dimensional output [0,0,1],[0,1,0],[1,0,0]

#Randomly sample from a normal distribution with standard deviation .01
weights = tf.Variable(tf.random_normal([numFeatures,numLabels],
                                       mean=0,
                                       stddev=0.01,
                                       name="weights"))

bias = tf.Variable(tf.random_normal([1,numLabels],
                                    mean=0,
                                    stddev=0.01,
                                    name="bias"))
##

## Logistic Regression Model ##
# Three-component breakdown of the Logistic Regression equation.
# Note that these feed into each other.
apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias") 
activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")
##


## TRAINING ##

## cost function ##
# Number of Epochs in our training
numEpochs = 700
# Defining our learning rate iterations (decay)
learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                          global_step= 1,
                                          decay_steps=trainX.shape[0],
                                          decay_rate= 0.95,
                                          staircase=True)
#Defining our cost function - Squared Mean Error
cost_OP = tf.nn.l2_loss(activation_OP-yGold, name="squared_error_cost")
##

#Defining our Gradient Descent
training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)

sess = tf.Session()
init_OP = tf.global_variables_initializer()
sess.run(init_OP)

'''
additional operations to keep track of our model's efficiency over time
'''
# argmax(activation_OP, 1) returns the label with the most probability
# argmax(yGold, 1) is the correct label
correct_predictions_OP = tf.equal(tf.argmax(activation_OP,1),tf.argmax(yGold,1))

# If every false prediction is 0 and every true prediction is 1, the average returns us the accuracy
accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))

# Summary op for regression output
activation_summary_OP = tf.summary.histogram("output", activation_OP)

# Summary op for accuracy
accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)

# Summary op for cost
cost_summary_OP = tf.summary.scalar("cost", cost_OP)

# Summary ops to check how variables (W, b) are updating after each iteration
weightSummary = tf.summary.histogram("weights", weights.eval(session=sess))
biasSummary = tf.summary.histogram("biases", bias.eval(session=sess))

# Merge all summaries
merged = tf.summary.merge([activation_summary_OP, accuracy_summary_OP, cost_summary_OP, weightSummary, biasSummary])

# Summary writer
writer = tf.summary.FileWriter("summary_logs", sess.graph)
##

## TRAINING LOOP ##
# Initialize reporting variables
cost = 0
diff = 1
epoch_values = []
accuracy_values = []
cost_values = []

# Training epochs
for i in range(numEpochs):
    if i > 1 and diff < .0001:
        print("change in cost %g; convergence."%diff)
        break
    else:
        # Run training step
        step = sess.run(training_OP, feed_dict={X: trainX, yGold: trainY})
        # Report occasional stats
        if i % 10 == 0:
            # Add epoch to epoch_values
            epoch_values.append(i)
            # Generate accuracy stats on test data
            train_accuracy, newCost = sess.run([accuracy_OP, cost_OP], feed_dict={X: trainX, yGold: trainY})
            # Add accuracy to live graphing variable
            accuracy_values.append(train_accuracy)
            # Add cost to live graphing variable
            cost_values.append(newCost)
            # Re-assign values for variables
            diff = abs(newCost - cost)
            cost = newCost

            #generate print statements
            print("step %d, training accuracy %g, cost %g, change in cost %g"%(i, train_accuracy, newCost, diff))

print()

# How well do we perform on held-out test data?
print("final accuracy on test set: %s" %str(sess.run(accuracy_OP, 
                                                     feed_dict={X: testX, 
                                                                yGold: testY})))

print()

# plot the cost to see how it behaves
plt.plot([np.mean(cost_values[i-50:i]) for i in range(len(cost_values))])
plt.show()
##
'''
Assuming no parameters were changed, you should reach a peak accuracy of 90% at the end of training,
which is commendable. Try changing the parameters such as the length of training,
and maybe some operations to see how the model behaves. Does it take much longer? How is the performance?
'''
