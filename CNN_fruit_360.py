import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

fruit_images = []
fruit_labels = []

for fruit_dir in glob.glob("data/Training/*"):
      fruit_label = fruit_dir.split("\\")[-1]
      fruit_labels.append(fruit_label)
      for fruit_images_path in glob.glob(fruit_dir + "/*"):
            image = mpimg.imread(fruit_images_path)
            fruit_images.append(image)



