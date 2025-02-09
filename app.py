from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import urllib.request
import os

app = Flask(__name__)
model = load_model(r'D:\riceplant disease detection\rice_disease_vgg16 (1).h5')

classes = ['Leaf_Blight', 'BrownSpot', 'Leaf Smut']  # Define the class names

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize to match your model's input
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None

    if request.method == 'POST':
        try:
            if 'file' in request.files and request.files['file'].filename:  # Handle uploaded file
                image = request.files['file']
                image_path = os.path.join('uploads', image.filename)
                image.save(image_path)
            elif 'url' in request.form and request.form['url']:  # Handle image URL
                image_url = request.form['url']
                image_path = 'temp.jpg'
                urllib.request.urlretrieve(image_url, image_path)
            else:
                error = "No image input provided."
                return render_template('index.html', prediction=prediction, error=error)

            # Preprocess and predict
            img_array = preprocess_image(image_path)
            prediction = model.predict(img_array)
            os.remove(image_path)  # Clean up temporary files

            # Get the predicted class
            predicted_class = classes[np.argmax(prediction)]
            prediction = f"Disease: {predicted_class}, Confidence: {np.max(prediction):.2f}"
        except Exception as e:
            error = f"An error occurred: {str(e)}"

    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
