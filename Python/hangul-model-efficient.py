import io
import os

import pandas as pd
import numpy as np

from keras.models import load_model, Model
 
from keras.preprocessing.image import img_to_array, load_img
from keras.utils.np_utils import to_categorical   
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import losses
from keras import optimizers

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.core.framework import graph_pb2

NO_CLASSES = 60   # Number of Hangul characters we are classifying
BATCH_SIZE = 50		# Size of each training batch
EPOCHS = 14				# Number of epochs to run
IMAGE_WIDTH = 64		# Width of each character in pixels
IMAGE_HEIGHT = 64		# Height of each character in pixels
TRAIN_TEST_SPLIT = 0.20 # Percentage of images to use in testing. Rest is used in testing

# Parameters
label_file = 'labels/{}-common-hangul.txt'.format(NO_CLASSES)
image_data_csv = 'image-data/labels-map.csv'
MODEL_NAME = 'hangul_convnet'

#return list of image file paths and corresponding labels
def files_labels():
    filenames = []
    labels = []
    with open(image_data_csv, 'r', encoding='utf-8') as csvfile:
        for line in csvfile:
            splitline = line.replace("\n", "").split(',')
            filenames.append(splitline[0])
            labels.append(splitline[1])
    #convert labels into one hot
    labels = np.array(labels)
    _, indices = np.unique(labels, return_inverse=True)
    labels = to_categorical(indices)
    return filenames, labels

# Get hangul image jpeg, convert to numpy array and return array
def get_img_as_array(path):
	img_jpeg = load_img(path, grayscale = True)
	img_array = img_to_array(img_jpeg)/255
	return img_array

#get a list of image paths and return list of image array
def load_images(filenames):
    images = np.array([get_img_as_array(filename) for filename in filenames])
    return images

#generate batches of images for training
def generator(filenames, labels):
    amount = len(filenames)
    while True:
        batch_start = 0
        batch_end = BATCH_SIZE

        while batch_start < amount:
            limit = min(batch_end, amount)
            X = load_images(filenames[batch_start: limit])
            Y = labels[batch_start:batch_end]
            #
            yield (X,Y)
            batch_start += BATCH_SIZE   
            batch_end += BATCH_SIZE

# Function copied: www.github.com/llSourcell/A_Guide_to_Running_Tensorflow_Models_on_Android/blob/master/tensorflow_model/mnist_convnet_keras.py
# Includes semi-original code to remove dropout layer for inference on mobile: https://dato.ml/drop-dropout-from-frozen-model/
def export_model(saver, model, input_node_names, output_node_name):

    tf.train.write_graph(K.get_session().graph_def, 'out', \
        MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None, \
        False, 'out/' + MODEL_NAME + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
    	input_graph_def.ParseFromString(f.read())

	# print nodes in graph, locate the connections into and out of dropout layer
	# for me, dropout starts at [41], ends at [64]
	# this will be different for any other graph 
    # display_nodes(input_graph_def.node)

	# connect '[68] dense_2/MatMul ' to '[41] dense_1/Relu ', remove dropout layer by connecting either end together
    input_graph_def.node[68].input[0] = 'dense_1/Relu'
    nodes = input_graph_def.node[:42] + input_graph_def.node[64:] 

	# create new GraphDef using our modified graph with no dropout layer
    input_graph_no_dropout = graph_pb2.GraphDef()
    input_graph_no_dropout.node.extend(nodes)

	# displaying the nodes we can see how the dropout layer has been removed entirely
    # display_nodes(input_graph_no_dropout.node)

	# optimise modified graph for inference
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_no_dropout, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

	# write to protobuff
    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("graph saved!")


#define and run the model
def run(x_train, x_test, y_train, y_test):
	# Define Model
    model = Sequential() 
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(64, 64, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))

    model.add(Flatten())  # flattens feature map to 1-d tensor
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5)) # removed to see if it works in android without it

    model.add(Dense(NO_CLASSES, activation = 'softmax')) # classification layer

    model.compile(loss=losses.categorical_crossentropy, 
    optimizer=optimizers.Adadelta(),
    metrics=['accuracy'])

    model.fit_generator( generator(x_train, y_train),
    steps_per_epoch=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0) # evaluate model accuracy
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return model # return model for saving


#get image directories and associated labels from csv
filenames, labels = files_labels()

#split data into training and testing
x_train, x_test_raw, y_train, y_test = train_test_split(filenames, labels, test_size=TRAIN_TEST_SPLIT)

#load test images to RAM
x_test = np.zeros((len(x_test_raw), 64, 64, 1))
for i in range(len(x_test_raw)):
    x_test[i] = get_img_as_array(x_test_raw[i])

#run model
model = run(x_train, x_test, y_train, y_test)

#save model as h5
model.save('hangul_model_{}_characters.h5'.format(NO_CLASSES))

# K.set_learning_phase(0)

# save model as .pb for use on android
# export_model(tf.train.Saver(), model, ["sequential_1_input"], "dense_2/Softmax")