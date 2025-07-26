from flask import Flask, jsonify, request, render_template, redirect, url_for
import time
from datetime import datetime
import os
from get_features import get_features
from keras.models import load_model
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)
is_monitoring = False

# or manually (make sure the order matches training)
emotion_classes = ['angry', 'calm', 'disgust', 'fear', 'happy', 'nepali', 'neutral', 'sad', 'scream', 'surprise']
model = load_model("emotion_recognizer.keras")

def reading_audio_files():

    file_path = []
    path = 'recordings'
    nepali_dataset = os.listdir(path)
    for file in nepali_dataset:
        file_path.append(path + '/' + file )

    return file_path

def prediction(audio_path):
    try:
        # featurees extraction 
        features = get_features(audio_path)
        features = np.array(features)

        # standard scaler transforming 
        scaler = joblib.load('standard_scaler.pkl')
        features_scaled = scaler.transform(features)
        features_scaled = np.expand_dims(features_scaled, axis=2)

        df = pd.DataFrame(emotion_classes, columns=['Emotions'])
        y_pred = model.predict(features_scaled)
        y_pred_index = emotion_classes[np.mean(np.argmax(y_pred, axis=1)).astype('int')]

        return y_pred_index
        
    except Exception as e:
        print("Error during prediction:", e)

@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template("home.html")

@app.route('/start-monitoring', methods=['GET', 'POST'])
def start_monitoring():
    global is_monitoring

    if request.method == 'POST':
        if not is_monitoring:
            is_monitoring = True
            
            from voice_recording import main

            print("voice recording started ...")
            main()
            print("voice recording stopped...")

            files_path = reading_audio_files()
            if files_path:
                predictions = []  
                for file in files_path:
                    pred = prediction(file)
                    if pred: 
                        predictions.append(pred)
                        print(f"File: {file}, Prediction: {pred}")
                    
                if predictions:
                    final_pred = predictions[-1] 

        return render_template("start_monitoring.html", values = final_pred, status='completed')
    else:
        return render_template("start_monitoring.html", values = None, status='starting')

@app.route('/stop-monitoring', methods=['POST'])
def stop_monitoring():
    
    from voice_recording import stop_recording 
    stop_recording()
    return {'status': 'stopped', 'message': 'voice monitoring stopped'}

@app.route('/emergency-call', methods=['POST'])
def emergency_call():
    print(" Manual emerrgency call triggered")
    return {'status': 'emergency_called', 'message':'Emergency service called'}

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0",  port=5050)