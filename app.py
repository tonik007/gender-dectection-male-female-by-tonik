import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load face and gender models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gender_model = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')

# Gender labels
GENDER_LIST = ['Male', 'Female']

def detect_gender(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    results = []

    for (x, y, w, h) in faces:
        face_roi = img[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (227, 227))
        
        # Preprocess for gender detection
        blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        gender_model.setInput(blob)
        gender_preds = gender_model.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]
        
        results.append({
            'box': [int(x), int(y), int(w), int(h)],
            'gender': gender
        })

    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Detect gender
            results = detect_gender(filepath)

            return render_template('index.html', filename=filename, results=results)
            
    return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)

    if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
