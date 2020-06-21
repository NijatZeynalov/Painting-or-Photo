#import modules

import os
import zipfile
import random
import shutil
from shutil import copyfile
from os import getcwd
import numpy as np

# make directions

os.mkdir('photos-v-paintings')
os.mkdir('photos-v-paintings/training')
os.mkdir('photos-v-paintings/testing')
os.mkdir('photos-v-paintings/training/photos')
os.mkdir('photos-v-paintings/training/paintings')
os.mkdir('photos-v-paintings/testing/photos')
os.mkdir('photos-v-paintings/testing/paintings')
print('done')

#split data to training and test list

def data_split(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    file_contents = os.listdir(SOURCE)
    files = []
    for i in range(len(file_contents)):
        if (os.path.getsize(SOURCE+file_contents[i])==1):
            print(file_contents[i]+" is ignoring, because it has zero length..")
        else:
            files.append(file_contents[i])
    training_length = int(len(files)*SPLIT_SIZE)
    testing_length = int(len(files)-training_length)
    shuffled_list = random.sample(files, len(files))
    training_list = shuffled_list[0:training_length]
    testing_list = shuffled_list[-testing_length:]
    for file in training_list:
        copyfile(SOURCE+file, TRAINING+file)
    for file in testing_list:
        copyfile(SOURCE + file, TESTING + file)

PAINTING_SOURCE_DIR = "photodata\\painting\\"
TRAINING_PAINTING_DIR = "photos-v-paintings\\training\\paintings\\"
TESTING_PAINTING_DIR = "photos-v-paintings\\testing\\paintings\\"

PHOTOS_SOURCE_DIR = 'photodata\\photos\\'
TRAINING_PHOTOS_DIR = "photos-v-paintings\\training\\photos\\"
TESTING_PHOTOS_DIR = "photos-v-paintings\\testing\\photos\\"

split_size = .9
data_split(PAINTING_SOURCE_DIR, TRAINING_PAINTING_DIR, TESTING_PAINTING_DIR, split_size)
data_split(PHOTOS_SOURCE_DIR, TRAINING_PHOTOS_DIR, TESTING_PHOTOS_DIR, split_size)

