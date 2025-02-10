from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Ensure uploads folder exists
UPLOAD_FOLDER = os.path.join('uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['uploads'] = UPLOAD_FOLDER

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'diabetic_retinopathy_model.h5')
model = load_model(model_path)

# Print the model's summary for debugging
print("Model Summary:")
model.summary()


# Define class labels
class_labels = {
    0: "No Signs of Retinopathy",
    1: "Mild Retinopathy",
    2: "Moderate Retinopathy",
    3: "Severe Retinopathy",
    4: "Proliferative Retinopathy"
}

# Function to preprocess the uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the file to the uploads folder
        filename = secure_filename(file.filename)
        file_path = os.path.join(os.getcwd(), app.config['uploads'], filename)
        file.save(file_path)

        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Perform prediction
        prediction = model.predict(img_array)  # Prediction will be a probability distribution
        predicted_class = np.argmax(prediction)  # Get the index of the highest probability
        confidence = np.max(prediction) * 100  # Get the confidence score as a percentage

        # Get the class label
        result = class_labels[predicted_class]

        # Relative path for the image
        image_url = filename  # Only store the filename, not the folder

        # Render the result page with the prediction
        return render_template('result.html', diagnosis=result, confidence=f"{confidence:.2f}", image_name=image_url)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)