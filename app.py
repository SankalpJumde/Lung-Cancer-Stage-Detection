from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Load your trained model (.keras or .h5)
model = tf.keras.models.load_model(r"D:/Lung Cancer/Flask/model/lung_cancer_model_finalll.keras")

# Define image upload path
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return img_array.reshape(1, 224, 224, 3)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    name = request.form["name"]
    age = request.form["age"]
    gender = request.form["gender"]
    image_file = request.files["image"]

    if image_file:
        filename = secure_filename(image_file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(filepath)

        processed_image = preprocess_image(filepath)
        prediction = model.predict(processed_image)
        classes = ['Stage 1', 'Stage 2', 'Stagw 3']
        predicted_class = classes[np.argmax(prediction)]

        return render_template(
            "result.html",
            name=name,
            age=age,
            gender=gender,
            image_path=filepath,
            prediction=predicted_class
        )
    else:
        return "No image uploaded", 400

if __name__ == "__main__":
    app.run(debug=True)
