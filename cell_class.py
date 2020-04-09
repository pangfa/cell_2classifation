import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import pickle
import keras
from sklearn.utils import shuffle

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

sns.set(style='white', context='notebook',palette='deep')
read_file=open(r'C:\Users\sen\Desktop\train_data.pkl','rb')
read_file_1=open(r'C:\Users\sen\Desktop\test_data.pkl','rb')
(X_train,Y_train)=pickle.load(read_file)
(x_test,y_test) =pickle.load(read_file_1)
read_file.close()


print(X_train.shape)
print(type(X_train))
plt.imshow(X_train[0])
plt.show()
print(Y_train.shape)
print(type(Y_train))
g = sns.countplot(Y_train)
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
axes = axes.flatten()
for i in range(10):
    axes[i].imshow(X_train[i])
    axes[i].set_xticks([])
    axes[i].set_yticks([])
plt.tight_layout()
plt.show()

print('这十张图片的标签分别是：', Y_train[:10])
x_train = X_train/255.0
x_test = x_test/255.0
y_train = to_categorical(Y_train)
y_test = to_categorical(y_test)
print(x_train.shape)
print(type(x_train))
import keras
print(y_train.shape)
print(type(y_train))
print(x_test.shape)
print(x_train.shape)
print(y_test.shape)
print(y_train.shape)
x_train,y_train = shuffle(x_train,y_train)
x_test,y_test = shuffle(x_test,y_test)

random_seed = 2
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.1,random_state=random_seed)

from keras.applications.vgg16 import VGG16
con_base = VGG16(weights = 'imagenet',include_top = False,input_shape=(128,100,1))
print(con_base.summary())

model = Sequential()
#model.add(Conv2D(32,(3,3),activation="relu",input_shape=(200,256,3)))
#model.add(Conv2D(32,(3,3),activation="relu"))
#model.add(MaxPool2D(pool_size=(2,2)))
#model.add(Dropout(0.0025))
#model.add(Conv2D(64,(3,3),activation="relu"))
#model.add(Conv2D(64,(3,3),activation="relu"))
#model.add(MaxPool2D(pool_size=(2,2)))
#model.add(Dropout(0.0025))
#model.add(Conv2D(128,(3,3),activation="relu"))
#model.add(Conv2D(128,(3,3),activation="relu"))
#model.add(MaxPool2D(pool_size=(2,2)))
#model.add(Dropout(0.0025))
model.add(con_base)
model.add(Flatten())
#model.add(Dense(512,activation="relu"))
model.add(Dense(256,activation="relu"))
model.add(Dense(128,activation="relu"))
#model.add(Dense(64,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4,activation="softmax"))
con_base.trainable = True
set_trainable = False
for layer in con_base.layers:
    if layer.name == 'block_conv1':
        set_trainable=True
    if set_trainable:
        layer.trainable=True
    else:
        layer.trainable=False

optimizer = RMSprop(lr=0.0001,rho=0.9,decay=0.0)
model.compile(loss="categorical_crossentropy",optimizer=optimizer,
             metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',patience=3,verbose=1,factor=0.5,min_lr=0.00001)
epochs = 50
batch_size = 16

datagen = ImageDataGenerator(featurewise_center=False,samplewise_center=False,
                             featurewise_std_normalization=False,samplewise_std_normalization=False,
                             zca_whitening=False,rotation_range=20,
                             zoom_range=0.2,width_shift_range=0.2,
                             height_shift_range=0.2,horizontal_flip=True,
                             vertical_flip=False)
datagen.fit(x_train)
history = model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),
                              epochs=epochs,validation_data=(x_val,y_val),
                              verbose=2,steps_per_epoch=x_train.shape[0]//batch_size,
                              callbacks=[learning_rate_reduction])

fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)
plt.show()

score = model.evaluate(x_test,y_test,verbose=0)
print('loss: ',score[0])
print('accuracy:',score[1])
