'''
In a simple linear regression there are two variables, the dependent variable, which can be seen as
the "state" or "final goal" that we study and try to predict, and the independent variables,
also known as explanatory variables, which can be seen as the "causes" of the "states".

When more than one independent variable is present the process is called multiple linear regression
When multiple dependent variables are predicted the process is known as multivariate linear regression

Y = aX + b;
Where Y is the dependent variable and X is the independent variable,
and a and b being the parameters we adjust.
a is known as "slope" or "gradient" and b is the "intercept"
'''
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import pylab as pl
import numpy as np
import tensorflow as tf

plt.rcParams['figure.figsize'] = (10,6)
X = np.arange(0.0, 5.0, 0.1)
print(X)
print()
##You can adjust the slope and intercept to verify the changes in the graph
a = 1
b = 0

Y= a * X + b 

plt.plot(X, Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

## Linear Regression w\ TensorFlow
'''
We have downloaded a fuel consumption dataset, FuelConsumption.csv, which contains model-specific
fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles
for retail sale in Canada.

MODELYEAR e.g. 2014
MAKE e.g. Acura
MODEL e.g. ILX
VEHICLE CLASS e.g. SUV
ENGINE SIZE e.g. 4.7
CYLINDERS e.g 6
TRANSMISSION e.g. A6
FUEL CONSUMPTION in CITY(L/100 km) e.g. 9.9
FUEL CONSUMPTION in HWY (L/100 km) e.g. 8.9
FUEL CONSUMPTION COMB (L/100 km) e.g. 9.2
CO2 EMISSIONS (g/km) e.g. 182 --> low --> 0
'''
df = pd.read_csv("FuelConsumptionCo2.csv")
print(df.head)
print()

train_x = np.asanyarray(df[['ENGINESIZE']])
train_y = np.asanyarray(df[['CO2EMISSIONS']])

# initialize with random a & b, define linear function
a = tf.Variable(20.0)
b = tf.Variable(30.2)
y = a * train_x + b

# define loss function
'''
we are going to define a loss function for our regression, so we can train our model to better fit our data.
In a linear regression, we minimize the squared error of the difference between the
predicted values(obtained from the equation) and the target values (the data that we have)

tf.reduce_mean() ==> finds the mean of a multidimensional tensor, and the result can have a different dimension
'''
loss = tf.reduce_mean(tf.square(y - train_y))

# define optimizer method
'''
The gradient Descent optimizer takes in parameter: learning rate, which corresponds to the speed
with which the optimizer should learn

We will use the .minimize() which will minimize the error function of our optimizer, resulting in a better model
'''
optimizer = tf.train.GradientDescentOptimizer(0.05)
train = optimizer.minimize(loss)

# init variables and run graph
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

loss_values = []
train_data = []
for step in range(100):
    _, loss_val, a_val, b_val = sess.run([train, loss, a, b])
    loss_values.append(loss_val)
    if step % 5 == 0:
        print(step, loss_val, a_val, b_val)
        train_data.append([a_val, b_val])

print()

# plot the loss values to see how it has changed during the training
plt.plot(loss_values, 'ro')
plt.show()

# visualize how the coefficient and intercept of line has changed to fit the data
cr, cg, cb = (1.0, 1.0, 0.0)
for f in train_data:
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)
    if cb > 1.0: cb = 1.0
    if cg < 0.0: cg = 0.0
    [a, b] = f
    f_y = np.vectorize(lambda x: a*x + b)(train_x)
    line = plt.plot(train_x, f_y)
    plt.setp(line, color=(cr,cg,cb))

plt.plot(train_x, train_y, 'ro')
green_line = mpatches.Patch(color='red', label='Data Points')
plt.legend(handles=[green_line])
plt.show()
