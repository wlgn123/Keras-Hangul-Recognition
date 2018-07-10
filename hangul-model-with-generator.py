import io
import pandas as pd
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.utils.np_utils import to_categorical   
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as bk
from keras import losses
from keras import optimizers

bk.tensorflow_backend._get_available_gpus()

# Parameters
label_file = "labels/2350-common-hangul.txt"
image_data = "image-data/labels-map.csv"

NO_CLASSES = 2350		# Number of Hangul characters we are classifying
BATCH_SIZE = 100		# Size of each training batch
EPOCHS = 50				# Number of epochs to run
IMAGE_WIDTH = 64		# Width of each character in pixels
IMAGE_HEIGHT = 64		# Height of each character in pixels
TRAIN_TEST_SPLIT = 0.75 # Percentage of images to use in training. Rest is used in testing


# Loads csv and image paths 
def get_dataset():
	# load data into local numpy arrays
	data = pd.read_csv(image_data, header = None, encoding = "utf-8").values # read in image-label mapping file with no header
	allLabels = io.open(label_file, 'r', encoding='utf-8').read().splitlines() # get all possible labels 
	features, labels = data[:, :-1], data[:, -1]
	
	# loop through all label, replace with int representing its indice position with corresponding char in allLabels
	count = 0
	#for i in range(len(label)):
	for i in range(len(labels)):
		if labels[i] == allLabels[count]:
			labels[i] = count
		else: # if label does not match, it will match the next character so we can make it equal to count+1
			labels[i] = count + 1			 
			count += 1

	return features, labels

# Split complete dataset according to TRAIN_TEST_SPLIT parameter
def split_dataset(features, labels):
	training = int(round(TRAIN_TEST_SPLIT * len(features))) # no. of features to use for training
	testing = int(len(features) - training)	# no. of features to use for testing

	# Create arrays to hold data
	x_train = np.empty((training, 1), dtype = "S100")
	y_train = np.zeros((training, 1))
	x_test = np.empty((testing, 1), dtype = "S100")
	y_test = np.zeros((testing, 1))

	# split data... 
	counter = 0
	for i in range(len(features)):
		if i < training:
			x_train[i] = features[i][0]
			y_train[i] = labels[i]
		else:
			x_test[counter] = features[i][0]
			y_test[counter] = labels[i]
			counter += 1
	return x_train, y_train, x_test, y_test

# Get hangul image jpeg, convert to numpy array 
def get_img_as_array(path):
    img_jpeg = load_img(path, grayscale = True)
    img_array = img_to_array(img_jpeg) / 255.0
    return img_array

# Gets batches of features and labels, converts JPEG to numpy and label to one_hot
def generator(features, labels, batch_size):
 	# Create empty arrays to contain batch of features and labels
	batch_features = np.zeros((batch_size, 64, 64, 1))
	batch_labels = np.zeros((batch_size, NO_CLASSES))
	while True:
	  	for i in range(batch_size):
		    # choose random index in features
		    index = np.random.choice(len(features),1)
		    batch_features[i] = get_img_as_array(features[index][0][0])
		    batch_labels[i] = to_categorical(labels[index], num_classes=NO_CLASSES); 
	yield batch_features, batch_labels

# Define the CNN
def main():

	features, labels = get_dataset()
	x_train, y_train, x_test, y_test = split_dataset(features, labels) # get training to feed to generator
	train_generator = generator(x_train, y_train, batch_size=100)
	test_generator = generator(x_test, y_test, batch_size=100)
 	
 	# Define Model 
	model = Sequential() 

	model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), 
	               		activation='relu'))
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

	model.fit_generator(generator = train_generator,
						validation_data = test_generator,
        				steps_per_epoch=1000,
        				epochs=10)

	#score = model.evaluate(x_test, y_test, verbose=0)
	#print('Test loss:', score[0])
	#print('Test accuracy:', score[1])

if __name__ == '__main__':
	main()
	
 	



