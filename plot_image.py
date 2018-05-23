import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import skimage.measure
import numpy as np

import pandas as pd
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

image_path = "data/Training/Orange/r_320_100.jpg"
image = mpimg.imread(image_path)
max_pool = []
max_pool.append(image)
for i in range(1,8):
    max_pool.append(skimage.measure.block_reduce(max_pool[i-1], (2,2,1), np.max))

fig = plt.figure(figsize=(8,8))
for i in range(1,5):
    fig.add_subplot(4,4,i)
    plt.imshow(max_pool[i-1])

image2 = cv2.imread(image_path, cv2.IMREAD_COLOR)
print(image2.shape[2])
cv = []
cv.append(image2)
for i in range(1,5):
    width = int(cv[i-1].shape[0]/2)
    length = int(cv[i-1].shape[1]/2)
    print(width, length)
    cv.append(cv2.resize(cv[i-1], (width,length)))
    print(cv[i].shape)

for i in range(1,5):
    fig.add_subplot(4,4,i+8)
    plt.imshow(cv[i-1])
plt.show()