
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model
model = load_model("model_cnn.h5")

# Prediction function
def predict_image(images_path):
    img = cv2.imread(images_path)
    img = cv2.resize(img,(224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return "Cat" if prediction[0][0] > 0.5 else "Dog"

# Routes
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    file = request.files['file']
    filepath = "uploaded.jpg"  # save inside static folder
    file.save(filepath)
    result = predict_image(filepath)
    return render_template("predict.html", prediction=result, image_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)