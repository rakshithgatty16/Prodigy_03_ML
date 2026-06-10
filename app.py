from flask import Flask, render_template, request
import cv2
import numpy as np
import pickle
from skimage.feature import hog

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

def features(img):
    img = cv2.resize(img,(64,64))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    return hog(
        gray,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        feature_vector=True
    )

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    feat = features(img)

    prediction = model.predict([feat])[0]

    result = "🐱 Cat" if prediction == 0 else "🐶 Dog"

    return render_template(
        "index.html",
        result=result
    )

if __name__ == "__main__":
    app.run(debug=True)