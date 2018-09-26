#Convolutional Neural Network

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialising the CNN
model = Sequential()

#Adding in the convolutional layer
model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

#adding in the pooling, reducing size of feature maps
model.add(MaxPooling2D(2,2))

model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(2,2))


#converting feature arrays to one flattened vector
model.add(Flatten())

#adding an artificial neural network to process the flattened data
model.add(Dense(output_dim = 128, activation = 'relu'))
model.add(Dense(output_dim = 1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#processing our images

from keras.preprocessing.image import ImageDataGenerator

#this rescales all of our pixels based on defined parameters to a number between 0 and 1
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

######################### Testing on a single picture #############

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog1.jpg', target_size = (64, 64))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

training_set.class_indices
result = model.predict(test_image)

if (result[0][0] == 1):
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
