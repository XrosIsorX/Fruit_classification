import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def plot_image(image, label):
      plt.figure(figsize=[5,5])
      plt.imshow(image)
      plt.title(label)
      plt.show()
      
dictionary = {}
number = 0
      
def class_to_number(c):
      n = []
      for i in c:
            n.append(dictionary[i])
      return np.array(n)
      
fruit_labels = []

train_x = []
train_y = []

valid_x = []
valid_y = []

count = 0
loop_count = 0

for fruit_dir in glob.glob("data/Training/*"):
      fruit_label = fruit_dir.split("\\")[-1]
      fruit_labels.append(fruit_label)
      if fruit_label not in dictionary:
            dictionary[fruit_label] = number
            dictionary[number] = fruit_label
            number+=1

train_labels = fruit_labels

for label in train_labels:         
      print(label)
      for fruit_dir in glob.glob("data/Training/" + label + "/*"):
            train_y.append(label)
            image = mpimg.imread(fruit_dir)
            image = cv2.resize(image, (75,75)))
            train_x.append(image)

for label in train_labels:         
      print(label)
      for fruit_dir in glob.glob("data/Validation/" + label + "/*"):
            valid_y.append(label)
            image = mpimg.imread(fruit_dir)
            image = cv2.resize(image, (75,75)))
            valid_x.append(image)

import numpy as np
from keras.utils import to_categorical

train_x = np.array(train_x)
train_y = train_y + fruit_labels
train_y = np.array(train_y)
# np.save('fruit_images', train_x)
# np.save('fruit_labels', train_y)

# train_x = np.load('fruit_images.npy')
# train_y = np.load('fruit_labels.npy')
train_y = class_to_number(train_y)
train_y = to_categorical(train_y)
train_y = train_y[:-len(fruit_labels)]

valid_x = np.array(valid_x)

valid_y = valid_y + fruit_labels
valid_y = np.array(valid_y)
valid_y = class_to_number(valid_y)
valid_y = to_categorical(valid_y)
valid_y = valid_y[:-len(fruit_labels)]


train_x = train_x.astype('float32')
train_x = train_x / 255

import keras 
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

batch_size = 64
epochs = 10
num_classes = 60

#Add network
model = Sequential()
model.add(Dense(128, activation='linear'), input_shape=(100,100,3))
model.add(LeakyReLU(alpha=0.0))
model.add(Dropout(0.1))              
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

model.summary()

# from keras.models import load_model

# model = load_model("fruit_model.h5py")

#Train network
model_train = model.fit(train_x, train_y, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_x, valid_y))
model.save("fruit_model.h5py")
