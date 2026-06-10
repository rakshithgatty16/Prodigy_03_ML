import os
import cv2
import pickle
from sklearn.svm import SVC
from skimage.feature import hog

X = []
y = []

def features(img):
    img = cv2.resize(img, (64,64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return hog(
        gray,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        feature_vector=True
    )

# Cats
for file in os.listdir("dataset/cats"):
    path = os.path.join("dataset/cats", file)

    img = cv2.imread(path)

    if img is None:
        continue

    X.append(features(img))
    y.append(0)

# Dogs
for file in os.listdir("dataset/dogs"):
    path = os.path.join("dataset/dogs", file)

    img = cv2.imread(path)

    if img is None:
        continue

    X.append(features(img))
    y.append(1)

print("Training...")

model = SVC(kernel="linear")

model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("model.pkl created successfully")