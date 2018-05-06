import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def plot_image(image, label):
      plt.figure(figsize=[5,5])
      plt.imshow(image)
      plt.title(label)
      plt.show()

fruit_images = []
fruit_labels = []
count = 0
loop_count = 0

# for fruit_dir in glob.glob("data/Training/*"):
#       fruit_label = fruit_dir.split("\\")[-1]
#       print(fruit_label)
#       for fruit_images_path in glob.glob(fruit_dir + "/*"):
#             fruit_labels.append(fruit_label)
#             image = mpimg.imread(fruit_images_path)
#             fruit_images.append(image)

# import numpy as np

# train_x = np.array(fruit_images)
# train_y = np.array(fruit_labels)
# np.save('fruit_images', train_x)
# np.save('fruit_labels', train_y)

train_x = np.load('fruit_images.npy')
train_y = np.load('fruit_labels.npy')

train_x = train_x.astype('float32')
train_x = train_x / 255


