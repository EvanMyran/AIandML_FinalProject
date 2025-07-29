#IMPORTS
#system set up
import sys

assert sys.version_info >= (3, 7)

from packaging import version
import sklearn

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

import matplotlib.pyplot as plt

#pretty plots!!!!
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

import numpy as np 
assert version.parse(np.__version__) >= version.parse("1.22.0")

#Tensorflow imports
import pandas as pd
import pathlib
from pathlib import Path
from matplotlib.pyplot import imread
from IPython.display import Image
import matplotlib.image as mpimg
import keras
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.utils import image_dataset_from_directory

# Imports for Deep Learning
from keras import layers
from keras.layers import Conv2D, Dense, Dropout, Flatten, InputLayer
from keras.models import Sequential

from keras.applications.resnet50 import preprocess_input, decode_predictions

from keras.utils import to_categorical

#variables for the directories of our train and test sets
#the basis for our training and valid set is the train_dir
train_dir = "../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/" # A/A1.jpg
test_dir = "../input/asl-alphabet-test/asl-alphabet-test"

#create our train, valid, and test sets from our imported data
train_data = image_dataset_from_directory(train_dir, labels='inferred', image_size=(200,200), seed=123, validation_split=.2, subset='training')
valid_data = image_dataset_from_directory(train_dir, labels='inferred', image_size=(200,200), seed=123, validation_split=.2, subset='validation')
test_data = image_dataset_from_directory(test_dir, labels='inferred', image_size=(200,200), seed=123)

# TEST SET
#break our test_data into x and y values
for images, labels in test_data:
    X_test = images.numpy()
    y_test = labels.numpy()
    break
print(X_test.shape)
y_test

# TRAIN SET
#break our test_data into x and y values
for images, labels in train_data:
    X_train = images.numpy()
    y_train = labels.numpy()
    break
print(X_train.shape)

class_names = train_data.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_data.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

#Sequential model
model = Sequential()
#input layer, define shape
model.add(layers.Input(shape=(200,200,3)))
#layer to rescale our images
model.add(layers.Rescaling(1./255)),
#convultion layer
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
#pooling layer for the images
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(29, activation='softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

#print summary of our model and layers
model.summary()

history = model.fit(train_data, epochs=10, validation_data=valid_data)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

import matplotlib.image as mpimg


#array of all images in colab with corresponding label
img_array = [
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/A_test.jpg", 'A'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/B_test.jpg", 'B'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/C_test.jpg",'C'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/D_test.jpg", 'D'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/E_test.jpg", 'E'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/F_test.jpg", 'F'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/G_test.jpg", 'G'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/H_test.jpg", 'H'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/I_test.jpg", 'I'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/J_test.jpg", 'J'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/K_test.jpg", 'K'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/L_test.jpg", 'L'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/M_test.jpg", 'M'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/N_test.jpg", 'N'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/O_test.jpg", 'O'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/P_test.jpg", 'P'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/Q_test.jpg", 'Q'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/R_test.jpg", 'R'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/S_test.jpg", 'S'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/T_test.jpg", 'T'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/U_test.jpg", 'U'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/V_test.jpg", 'V'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/W_test.jpg", 'W'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/X_test.jpg", 'X'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/Y_test.jpg", 'Y'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/Z_test.jpg", 'Z'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/nothing_test.jpg", 'nothing'],
    ["../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/space_test.jpg", 'space']]


#for loop to iterate through image name array and print predictions
for i, label in img_array:
    test_path = i
    img = keras.utils.load_img(test_path, target_size=(200, 200))
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # model makes prediction on the image
    preds = model.predict(x)

    #get the predicted class index
    predicted_class_index = np.argmax(preds[0])
    predicted_class_label = class_names[predicted_class_index]
    
    # displays the image
    image = mpimg.imread(test_path)
    plt.imshow(image)
    plt.title('Actual Class: '+label+' - Predicted Class: '+predicted_class_label)
    plt.axis('off')
    plt.show()
