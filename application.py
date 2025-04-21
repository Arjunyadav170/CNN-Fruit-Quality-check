
from flask import Flask, render_template, request, jsonify
import os
import pickle
import numpy as np
# from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

with open('model.pkl','rb') as f:
    model=pickle.load(f)

class_labels = ['apple', 'banana', 'bittergroud', 'capsicum', 'cucumber', 'okra',
                'oranges', 'potato', 'potato', 'tomto', 'tomato']

def load_image(img_path):
    img = image.load_img(img_path, target_size=(200, 200))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('fruit_image')

    if not file:
        return jsonify({'error': 'Please upload an image.'})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    x = load_image(filepath)
    output = model.predict(x)

    score1 = output[0][0][0]
    score2 = output[1][0]
    predict_label = np.argmax(score2)
    prediction_label_prob = class_labels[predict_label]
    freshness = "Fresh" if score1 > 0.5 else "Stale"

    return jsonify({
        'fruit': prediction_label_prob,
        'freshness': freshness,
        'image_url': f"/static/uploads/{file.filename}"
    })
if __name__ == '__main__':
    app.run(debug=True)
