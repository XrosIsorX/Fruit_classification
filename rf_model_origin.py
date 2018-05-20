import numpy as np
import pandas as pd
import cv2
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

fruit_images = []
labels = []
for fruit_dir_path in glob.glob("./data/fruits-360/Training/*"):
    fruit_label = fruit_dir_path.split("/")[-1]
    fruit_label = fruit_label.split("\\")[-1]
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (30, 30))
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        fruit_images.append(image)
        labels.append(fruit_label)
fruit_images = np.array(fruit_images)
labels = np.array(labels)
print(fruit_images.shape)

label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
print(label_to_id_dict)
label_id = np.array([label_to_id_dict[x] for x in labels])

scaler = StandardScaler()
images_scaled = scaler.fit_transform([i.flatten() for i in fruit_images])
print(images_scaled.shape)

pca = PCA(n_components=80)
pca_result = pca.fit_transform(images_scaled)
print(pca_result.shape)

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(pca_result, label_id, test_size=0.20, random_state=69)

rf = RandomForestClassifier(n_estimators=30, n_jobs=-1)
rf.fit(train_x, train_y)
predictions = rf.predict(test_x)
precision = accuracy_score(predictions, test_y) * 100
print("Accuracy with RandomForest: {0:.6f}".format(precision))
joblib.dump(rf, 'model_RF.pkl')

validation_fruit_images = []
validation_labels = []
for fruit_dir_path in glob.glob("./data/fruits-360/Validation/*"):
    fruit_label = fruit_dir_path.split("/")[-1]
    fruit_label = fruit_label.split("\\")[-1]
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        image = cv2.resize(image, (30, 30))
        validation_fruit_images.append(image)
        validation_labels.append(fruit_label)
validation_fruit_images = np.array(validation_fruit_images)
validation_labels = np.array(validation_labels)

validation_label_ids = np.array([label_to_id_dict[x] for x in validation_labels])
validation_images_scaled = scaler.transform([i.flatten() for i in validation_fruit_images])
validation_pca_result = pca.transform(validation_images_scaled)

test_predictions = rf.predict(validation_pca_result)
precision = accuracy_score(test_predictions, validation_label_ids) * 100
print("Validation Accuracy with Random Forest: {0:.6f}".format(precision))
