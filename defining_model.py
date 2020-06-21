#import libraries

import tensorflow as tf
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import numpy as np
from keras.layers import BatchNormalization

#define the model

layers = [
    tf.keras.layers.Conv2D(16, (3,3), activation = 'relu',padding='same',input_shape = (150, 150, 3)),
    tf.keras.layers.MaxPool2D(pool_size = (2,2), strides=(2,2)),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='same', kernel_initializer="he_normal" ),
    tf.keras.layers.MaxPool2D(pool_size = (2,2), strides=(2,2)),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same' , kernel_initializer="he_normal"),
    tf.keras.layers.MaxPool2D(pool_size = (2,2), strides=(2,2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal"),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal"),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
]
model = tf.keras.models.Sequential(layers)
#compile the model

model.compile(optimizer =Adam(lr=1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

#train and validation data generator

TRAIN_DIR = 'photos-v-paintings\\training'

train_datagen = ImageDataGenerator(
      rescale = 1./255,
	  rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(TRAIN_DIR, batch_size= 20, class_mode='binary', target_size=(150, 150))

VALIDATION_DIR = 'photos-v-paintings\\testing'
validation_datagen = ImageDataGenerator(rescale=1.0/255)
validation_generator = train_datagen.flow_from_directory(VALIDATION_DIR, batch_size=20, class_mode='binary', target_size=(150, 150))


#fit the model

history = model.fit_generator(train_generator, steps_per_epoch = 20,
            epochs = 25,
            validation_steps = 20, validation_data = validation_generator)

#save model weights

model.save_weights('my_model3.tf')

#plot model

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.show()
