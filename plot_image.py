import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import skimage.measure
import numpy as np

image = mpimg.imread("data/Training/Apple Braeburn/r_320_100.jpg")
max_pool = []
max_pool.append(image)
for i in range(1,8):
    max_pool.append(skimage.measure.block_reduce(max_pool[i-1], (2,2,1), np.max))

fig = plt.figure(figsize=(8,8))
for i in range(1,9):
    fig.add_subplot(2,4,i)
    plt.imshow(max_pool[i-1])
plt.show()