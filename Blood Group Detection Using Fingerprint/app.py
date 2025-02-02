from flask import Flask, request, render_template, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from PIL import Image
import os

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model_blood_group_detection.h5')

def preprocess_image(img):
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img = img.resize((256, 256))  # Adjust the size as per your model
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

blood_group_mapping = {0: 'A+', 1: 'A-', 2: 'AB+', 3: 'AB-', 4: 'B+', 5: 'B-', 6: 'O+', 7: 'O-'}

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            img = Image.open(file)  
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img = img.resize((256, 256))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Predict and map result to blood group
            result = model.predict(x)
            predicted_index = np.argmax(result, axis=1)[0]
            blood_group = blood_group_mapping[predicted_index]

            # Redirect to the result page with the prediction
            return redirect(url_for('display_result', blood_group=blood_group))
    return render_template('index.html')

@app.route('/result')
def display_result():
    blood_group = request.args.get('blood_group', None)
    return render_template('result.html', blood_group=blood_group)

if __name__ == '__main__':
    app.run(debug=True)