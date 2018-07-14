import io
import os

import pandas as pd
import numpy as np

from keras.preprocessing.image import img_to_array, load_img
from keras.utils.np_utils import to_categorical   
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import losses
from keras import optimizers

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


# Parameters
label_file = 'labels/10-common-hangul.txt'
image_data = 'image-data/labels-map.csv'
MODEL_NAME = 'hangul_convnet'

NO_CLASSES = 10		# Number of Hangul characters we are classifying
BATCH_SIZE = 30		# Size of each training batch
EPOCHS = 1				# Number of epochs to run
IMAGE_WIDTH = 64		# Width of each character in pixels
IMAGE_HEIGHT = 64		# Height of each character in pixels
TRAIN_TEST_SPLIT = 0.25 # Percentage of images to use in testing. Rest is used in testing


# Prepares features and labels arrays. Converts labels to onehot and jpegs to numpy arrays (memory intensive)
def create_dataset():
	data = pd.read_csv(image_data, header = None, encoding = "utf-8").values # read in image-label mapping file with no header
	allLabels = io.open(label_file, 'r', encoding='utf-8').read().splitlines() # get all possible labels 
	features, labels = data[:, :-1], data[:, -1]
	labels = data[:, -1]

	features_as_array = np.zeros((len(features), 64, 64, 1)) # image shape is 64 x 64 pixels, 1 channel

	# loop through all features labels, replace label with int representing its indice position with corresponding char in allLabels
	count = 0
	#for i in range(len(label)):
	for i in range(len(labels)):
		#print label[i] + " = " + allLabels[count] + "?"
		if labels[i] == allLabels[count]:
			labels[i] = count
			features_as_array[i] = get_img_as_array(features[i][0]) # Save image in our new array
		else: # if label does not match, it will match the next character so we can make it equal to count+1
			labels[i] = count + 1			 
			features_as_array[i] = get_img_as_array(features[i][0])
			count += 1
	# Convert int labels to one_hot labels using to_categorical
	one_hot_labels = to_categorical(labels, num_classes=len(allLabels))
	return features_as_array, one_hot_labels

# Function copied from github.com/llSourcell/A_Guide_to_Running_Tensorflow_Models_on_Android/blob/master/tensorflow_model/mnist_convnet_keras.py
def export_model(saver, model, input_node_names, output_node_name):

	#K.set_learning_phase(0)

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

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("graph saved!")

# Get hangul image jpeg, convert to numpy array and return array
def get_img_as_array(path):
	img_jpeg = load_img(path, grayscale = True)
	img_array = img_to_array(img_jpeg)/255
	return img_array

def print_model_layers(model):
	for n in tf.get_default_graph().as_graph_def().node:
		print(n.name) 

# Define the CNN
def main(x_train, x_test, y_train, y_test):

	# Define Model
	model = Sequential() 
	model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

	model.add(Conv2D(64, (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))

	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))

	model.add(Flatten())  # flattens feature map to 1-d tensor
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(NO_CLASSES, activation = 'softmax')) # classification layer

	model.compile(loss=losses.categorical_crossentropy, 
	optimizer=optimizers.Adadelta(),
	metrics=['accuracy'])

	model.fit(x_train, y_train, # train on train/testing data
	batch_size=BATCH_SIZE,
	epochs=EPOCHS,
	verbose=1,
	validation_data=(x_test, y_test))

	score = model.evaluate(x_test, y_test, verbose=0) # evaluate model accuracy
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	return model # return model for saving

if __name__ == '__main__':

	if not os.path.exists('out'):
			os.mkdir('out')

	features, labels = create_dataset() # generate training data

	x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=TRAIN_TEST_SPLIT) # split data in training/testing batches 

	model = main(x_train, x_test, y_train, y_test)

	K.set_learning_phase(0)

	export_model(tf.train.Saver(), model, ["sequential_1_input"], "dense_2/Softmax") # save model as .pb for use on android
	