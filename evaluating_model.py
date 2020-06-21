#import modules

import cv2
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt

#evaluate model

eval_model = tf.keras.Sequential(layers)
eval_model.load_weights('my_model3.tf')
path = 'ij.jpeg'
img = image.load_img(path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = eval_model.predict(images, batch_size=10)
print(classes[0])
prediction = ''
if classes[0]>0.5:
    prediction ='Painting'
else:
    prediction = 'Photo'

#plot the test results

plt.imshow(img)
plt.title(prediction)
plt.show()