#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense , Conv2D ,Dropout , Flatten , MaxPooling2D
# Dense for creating hidden and output layer 
#conv2D for  creating convolution network layers
#dropout to remove the random neurons
#Flatten used to convert input into vector
# maxpooling 2d to apply max pooling
import matplotlib.pyplot as plt # plot image

import numpy as np # numerical calculation to predict the class


# In[3]:


mnist = tf.keras.datasets.mnist 
# mnist is image classification dataset which contain images of numbers
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Splitting it into test and train sets
input_shape =(28,28 ,1)


# In[4]:


# shaping data into same dimension 
x_train = x_train.reshape(x_train.shape[0] , 28 ,28 ,1)
x_test = x_test.reshape(x_test.shape[0] , 28 ,28 ,1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# In[5]:


#normalizing between 0 to 1
x_train = x_train / 255  # Converting values from (0, 255) to (0, 1)
x_test = x_test / 255
print("Shape of training : " , x_train.shape)
print("Shape of training : " , x_test.shape)


# In[7]:


# creating model

model = Sequential() # we want to add one layer over another i.e. stack of layer hence used sequential
model.add(Conv2D(28 , kernel_size =(3,3) , input_shape= input_shape))
# 28 conv layer each of size 3x3 
model.add(MaxPooling2D(pool_size =(2,2))) # pooling size us size of filter or pool size
model.add(Flatten()) #convert into vector
model.add(Dense(200 , activation = "relu")) #hidden layer
model.add(Dropout(0.3)) # random neurons are removed like here 30% (0.3) are removed
model.add(Dense(10,activation = "softmax")) # output layer
model.summary()


# In[8]:


# training model
model.compile(optimizer='adam',loss="sparse_categorical_crossentropy", metrics =["accuracy"])
# here used adam as optimizer coz they have not mentioned any particular model
model.fit(x_train , y_train , epochs =2)


# In[9]:


#Estimating models performance
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Loss = %.3f" % test_loss)
print("Accuracy = %.3f" % test_acc)


# In[34]:


image=  x_train[0]
plt.imshow(np.squeeze(image), cmap ='gray')
#squeeze  remove items with single dimension from the array
#images are gray hence cmap = gray
plt.show()


# In[35]:


# predicting he class of images

image = image.reshape(1, image.shape[0],image.shape[1], image.shape[2])
predict_model = model.predict([image])
print("Predicted class :{}".format(np.argmax(predict_model)))
#argmax is used to give class with higher probability of prediction


# In[ ]:




