#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 22:51:58 2018

@author: ozgur
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import pandas as pd
import time

#%%
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#%%
#Note1
#Placeholders for features and labels. Placeholders stand for non-changeable 
#vector like objects that represents inputs. Here, feature placeholder can be 
#thougt as a 1x784 matrix version of flattened 28*28 picture and label 
#placeholder is 1x10 version of labels (because there are 10 labels). 
#Feature placeholder will be multiplied with 784xk weight matrix and 
#become [1,k] hidden layer where k refers number of node in that hidden layer.
X = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

n_hidden_layer1_nodes = 512
n_hidden_layer2_nodes = 256
#Hidden and output layer's weight vectors. Weights are normally distributed 
#at initial. Then they will be changed at each iteration. Therefore they are 
#created as tf.Variable.
h1_weights = tf.Variable(tf.truncated_normal([784, n_hidden_layer1_nodes]))
h2_weights = tf.Variable(tf.truncated_normal([n_hidden_layer1_nodes, n_hidden_layer2_nodes]))
out_weights = tf.Variable(tf.truncated_normal([n_hidden_layer2_nodes, 10]))

#Layer bias vectors with full of zeros, meaning that actually there won't 
#be bias in any layer.
h1_biases = tf.Variable(tf.zeros([n_hidden_layer1_nodes]))
h2_biases = tf.Variable(tf.zeros([n_hidden_layer2_nodes]))
out_biases = tf.Variable(tf.zeros([10]))

#%%
#Hidden layers can be add here with keeping in mind that they will be 
#multiplied wiht each other according to Note1. 
h1_layer = tf.matmul(X, h1_weights) + h1_biases
#At second hidden layer Relu activation function is used to reduce some nodes'
#significance before output layer. Frankly, it is an experimental op.
h2_layer = tf.nn.relu(tf.matmul(h1_layer, h2_weights) + h2_biases)
digit_weights = tf.matmul(h2_layer, out_weights) + out_biases

#%%
#A loss function wrt classification problem is defined here to be optimized. 
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=digit_weights, labels=y))

#Note2: It is surprising that tf can understands our nn structure and set 
#hidden layer weights at each iteration by just passing "digit_weights" tensor 
#in to loss function. What I mean is below code block does same too...
# =============================================================================
# def multi_layer(input_placeholder):
#     h1_layer = tf.matmul(X, h1_weights) + h1_biases
#     h2_layer = tf.nn.relu(tf.matmul(h1_layer, h2_weights) + h2_biases)
#     digit_weights = tf.matmul(h2_layer, out_weights) + out_biases
#     return digit_weights
# 
# logits = multi_layer(X)
# loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
# =============================================================================

#Here the optimizer function that optimzes loss function is defined. 
#Note3: In your example Gradient Descent optimizer is used with 
#learning rate of 0.5. Here I used AdamOptimizer wich is one of the most 
#popular optimizers in literature. But crucial point here is the learning rate. 
#Learning rate must be decided wrt optimizer and number of hidden layers. 
#For instance, in your single layer example, if 0.005 learning rate is used for 
#Gradient Descent, accuracy will be under 0.9. But if optimizer set to Adam 
#with 0.005 learning rate, accuracy will be above 0.9 again. However if we add 
#second hidden layer and then run Gradient Descent optimizer with 0.5 
#learning rate, accuracy will fall below 0.1 again.
optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
optimizer = optimizer.minimize(loss_function)

correct_prediction = tf.equal(tf.argmax(tf.nn.relu(digit_weights),axis=1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#%%
  

acc_table = pd.DataFrame()
time_table = pd.DataFrame()
num_epoch = [3,4,5,6,7]
batch_size = [50,100,150,300,500]

for epoch in num_epoch:
    print('Number of epochs is '+str(epoch))

    for size in batch_size:
        total_batch = int(mnist.train.num_examples/size)
        print('Batch size is '+str(size))
        start = time.time()
#I think starting a tf.Session like this is more easy to understand that 
#training process is started here.
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
#Note4: MNIST train dataset has 55000 training samples. At class example 100 samples is picked
#at each for 2000 iterations. That means class example passes over whole train dataset 
#2000/550~3.6 times (epochs). This code block passes over whole data exactly
#"num_epoch" times. 
            avg_cost = 0.
            for i in range(epoch):
                for x in range(total_batch):
                    current_batch = mnist.train.next_batch(size)
                    _, c = sess.run([optimizer,loss_function], feed_dict={X: current_batch[0], y: current_batch[1]})
                    avg_cost += c / total_batch
            acc=accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})        
            acc_table.at[size,epoch]=float("{0:.3f}".format(acc))
            print("Model Loss value: " + str(avg_cost))
            print("Test Accuracy: " + str("{0:.3f}".format(acc)))

        end = time.time()
        time_elapsed=end - start
        print('Seconds elapsed: '+str("{0:.2f}".format(time_elapsed)))
        print('---------------------')
        time_table.at[size,epoch]=float("{0:.2f}".format(time_elapsed))

    print('#####################')
    
#As for the question, adding another hidden layer slightly increase accuracy that 
#may be arguable when considering its performance.


# =============================================================================
# time_table
# Out[107]: 
#          3      4      5      6      7
# 50   11.15  16.10  24.38  22.15  25.58
# 100   7.54  12.72  12.56  15.88  17.85
# 150   6.36   9.67  10.79  13.04  14.65
# 300   5.12   7.41   9.10   9.95  11.64
# 500   4.85   8.41   7.31   8.12   9.60
# =============================================================================

#smaller batch_size takes more training time 
#increase in number of epochs increases training time

# =============================================================================
# acc_table
# Out[108]: 
#          3      4      5      6      7
# 50   0.929  0.906  0.755  0.789  0.830
# 100  0.956  0.951  0.955  0.954  0.953
# 150  0.940  0.955  0.959  0.962  0.964
# 300  0.940  0.949  0.952  0.953  0.956
# 500  0.938  0.942  0.948  0.952  0.948
# =============================================================================

#there is an optimum batch_size for the dataset, here it seems optimum batch size is 150 
#bigger number of epochs with optimum batch size, improve accuracy

#%%

fig = plt.figure(figsize=(16,9))
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

ax1.plot(acc_table.loc[50,:])
ax1.plot(acc_table.loc[100,:])
ax1.plot(acc_table.loc[150,:])
ax1.plot(acc_table.loc[300,:])
ax1.plot(acc_table.loc[500,:])

ax2.plot(time_table.loc[50,:])
ax2.plot(time_table.loc[100,:])
ax2.plot(time_table.loc[150,:])
ax2.plot(time_table.loc[300,:])
ax2.plot(time_table.loc[500,:])

fig.legend(loc='best')
plt.show()

#%%
#Same example with Keras including extra script for trying different optimizer,batch_size and epoch

from keras.models import Sequential #Sequential model is used for keras to do matrix multiplications of layers by itself.
from keras.layers import Dense, Activation, Dropout #Dense is the layer tensor, Activation is used for implementing activation function to the dense 
#and Dropout is used to reduce overfitting.
from keras.optimizers import Adam, SGD, RMSprop 
import random

X=mnist.train.images
y=mnist.train.labels
X_test=mnist.test.images[:100]
y_test=mnist.test.labels[:100]

model = Sequential([
#model takes input as tf.placeholder(shape=[None,784])
    Dense(512, input_shape=(784,)),
    Dropout(0.1),
    Dense(256),
    Activation('relu'),
    Dropout(0.1),
    Dense(10),
    Activation('softmax'),
])

for i in range(3):
    optimizers = {'Adam':Adam(lr=0.005), 'SGD':SGD(lr=0.5), 'RMSprop':RMSprop(lr=0.001)}
    optimizers= random.choice(list(optimizers.items()))
    batch_size = random.choice([100, 300, 500])
    num_epochs = random.choice([4,6,8])
    
    print('Optimizer:'+str(optimizers[0]), ',' , 'Batch size:'+str(batch_size), ',', 'Number of Epochs:'+str(num_epochs))
    
    model.compile(optimizer=optimizers[1],
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    
    model.fit(X, y, epochs=num_epochs, batch_size=batch_size)
    score = model.evaluate(X_test, y_test, batch_size=100)
    print('Model evaluation scores on 100 test images:')
    print(str(model.metrics_names[0])+':'+str(score[0]),',', str(model.metrics_names[1])+':'+str(score[1]))
    print('\n')

#Note5: Keras is easier to build compared to raw tf build. But to understand what a 
#nn does behind the scenes, one should get through every method/function/variable of the tf code before going into Keras 
