#%% IMPORT LIB BLCOK
import io
import pandas as pd
import numpy as np
from keras.preprocessing.image import img_to_array, load_img,ImageDataGenerator
from keras.utils.np_utils import to_categorical   
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

#%% MAIN BLOCK

NUM_CLASSES = 2412
IMG_SHAPE = (64,64,1)
# Loads csv and image paths 
def get_dataset(image_label_path = './image-data/labels-map.csv', label_path = './labels/2412-common-hangul-eng-num.txt'):
    if(image_label_path is None or label_path is None):
        print("IMAGE PATH OR LABEL PATH IS NONE")
        return

	# load data
    data = pd.read_csv(image_label_path, header = None, encoding = "utf-8")
    data.columns = ['path', 'label']
    allLabels = io.open(label_path, 'r', encoding='utf-8').read().splitlines() # get all possible labels 
    labels = data['label'].values
	
	# loop through all label, replace with int representing its indice position with corresponding char in allLabels
    count = 0
    #for i in range(len(label)):
    for i in range(len(labels)):
        if labels[i] == allLabels[count]:
            labels[i] = count
        else: # if label does not match, it will match the next character so we can make it equal to count+1
            labels[i] = count + 1			 
            count += 1

    data['label'] = labels
    # shuffle rows
    data = data.sample(frac=1).reset_index(drop=True).values    
    # x and y and data_size
    return data[:,:-1], to_categorical(data[:,-1:]), len(data)

# Get hangul image jpeg, convert to numpy array 
def get_img_as_array(path):
    img_jpeg = load_img(path, grayscale = True)
    img_array = img_to_array(img_jpeg) / 255
    return img_array

def get_generator(x_set, y_set, batch_size = 32):
    print(x_set.shape)
    print(y_set.shape)

    img_x_set = []

    for path in x_set:
        img_x_set.append(get_img_as_array(path[0]))

    img_x_set = np.asarray(img_x_set)

    print(img_x_set.shape)

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    datagen.fit(img_x_set)
    gen = datagen.flow(img_x_set, y_set)
    return gen

def create_model():
    # Define Model 
    model = Sequential()    
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=IMG_SHAPE))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same')) 
    
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same')) 
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same')) 

    model.add(Flatten())  # flattens feature map to 1-d tensor
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(NUM_CLASSES, activation = 'softmax')) # classification layer 
    
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model.summary()
    return model

#%% RUNNING BLOCK

if __name__ == "__main__":
    sample_size = 2000
    batch_size = 32

    x_set, y_set, data_size = get_dataset()
    model = create_model()

    for i in range(0, int(data_size / sample_size)-1):
        start_index = i*sample_size
        end_index = (i*sample_size) + sample_size

        if(end_index > data_size):
            end_index = data_size

        print("{}: {} ~ {}".format(i, start_index, end_index))
        val_index = int(end_index * 0.7)

        print("MAKE TRAIN GENERATOR")
        train_gen = get_generator(x_set[start_index:val_index], 
                                  y_set[start_index:val_index],
                                  batch_size = batch_size)

        print("MAKE VALID GENERATOR")
        valid_gen = get_generator(x_set[val_index:end_index],
                                    y_set[val_index:end_index])

        model.fit_generator(train_gen, validation_data=valid_gen,
                            steps_per_epoch=600, epochs=40, validation_steps=200)

        model.save("./save_point/model_{:0>4}.h5".format(i))
        model.save_weights('./save_point/model_weight_{:0>4}.h5'.format(i))
#%%
