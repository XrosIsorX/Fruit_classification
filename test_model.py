from keras.models import load_model

model = load_model("fruit_model_e20_all.h5py")

import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

fruit_labels = []

train_x = []
train_y = []

valid_x = []
valid_y = []

dictionary = {}
number = 0

import numpy as np
      
def class_to_number(c):
      n = []
      for i in c:
            n.append(dictionary[i])
      return np.array(n)

for fruit_dir in glob.glob("data/Training/*"):
      fruit_label = fruit_dir.split("\\")[-1]
      fruit_labels.append(fruit_label)
      if fruit_label not in dictionary:
            dictionary[fruit_label] = number
            dictionary[number] = fruit_label
            number+=1

train_labels = fruit_labels[:10]

for label in train_labels:         
      print(label)
      for fruit_dir in glob.glob("data/Validation/" + label + "/*"):
            valid_y.append(label)
            image = mpimg.imread(fruit_dir)
            valid_x.append(image)

from numpy import argmax

valid_x = np.array(valid_x)
result = model.predict(valid_x[100:110])
result = argmax(result, axis=1)
result = class_to_number(result)
print('valid_y',valid_y[100:110])
print('result ', result)
