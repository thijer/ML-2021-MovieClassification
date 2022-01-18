import numpy as np
import os
import csv
import cv2

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin")
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path = os.path.join('D:\\', 'Dataset', 'movie_posters', 'normal')

# range(len(os.listdir(os.chdir(path))))
# Import dataset
print("Start getting the data")
dataset = [cv2.imread(os.listdir(os.chdir(path))[x]) for x in range(15000)]
# Load the names of each picture
names = os.listdir(os.chdir(path))
data = np.asarray(dataset)
dataset_length = len(data)
del dataset

print("Done with importing data")


print("Start creating target arrays")
# Get the idx of each figure
for x in range(len(names)):
    names[x] = int(os.path.splitext(names[x])[0])

# Import labels for the set
path1 = os.path.join('D:\\', 'Dataset', 'movie_posters')
os.chdir(path1)
target_arrays = np.zeros(shape=(dataset_length, 13))
names = np.sort(names)

# Extract all the target arrays for each picture in the set
with open('labels.txt', encoding="utf8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    name_index = 0
    for row in csv_reader:
        if int(row[0]) == names[name_index]:
            target_arrays[name_index, :] = np.asarray(row[-16:-3])
            name_index += 1
        line_count += 1
        if name_index == dataset_length:
            break

del names

data = data[0:dataset_length, :, :, :]
target_arrays = target_arrays[0:dataset_length, :]
print("Finished creating target arrays")

# Make a train and test split
(train_dat, test_dat, train_lab, test_lab) \
        = train_test_split(data, target_arrays, test_size=0.20)

del data, target_arrays

# Specify height, width and dimensionality
width_input = 256   # width was 256 or 255 for all posters
height_input = 320  # Subject to change
dimensionality = 3  # colors are encoded in RGB, so dimensionality of each pixel is 3

model = Sequential()
# First conv layer
model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation="relu", input_shape=(height_input, width_input
                                                                               , dimensionality)))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Second conv layer
model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Third conv layer
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Flatten for input into dense layers
model.add(Flatten())
# First dense layer
model.add(Dense(768, activation='relu'))
model.add(Dropout(0.25))
# Second dense layer
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
# Output layer
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_dat, train_lab, epochs=10, validation_data=(test_dat, test_lab), batch_size=64)
