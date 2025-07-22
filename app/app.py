from flask import Flask, jsonify, request, render_template, redirect, url_for
import time
import queue 
import json 
import threading
from datetime import datetime
from voice_recording import audio_recorder

app = Flask(__name__)

is_monitoring = True


'''
    if trigger is detected then call trigger_emergency_logging function
    for logging datetime and .wav file into a json
    which will 
'''
def trigger_emergency_logging(detected_text):
    timestamp = datetime.now().isoformat()
    emergency = {
        'timestamp': timestamp,
        # i want to add detected file in a json file
        'status': 'emergency'
    }
    with open('voice_alerts.json', 'a') as f:
        data = json.dumps(emergency)
        f.write(data)

@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template("home.html")

@app.route('/start-monitoring', methods=['POST'])
def start_monitoring():
    global is_monitoring

    if not is_monitoring:
        is_monitoring = True

        # start background threads 
        # listen_thread = threading.Thread(target=listen_continuously, daemon=True)
        # process_thread = threading.Thread(target=process_voice, daemon=True)

        # listen_thread.start()
        # process_thread.start()

        return {'status': 'started', 'message':'voice monitoring started'}
    else:
        return {'status': 'already_running', 'message': 'already monitoring'}

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    global is_monitoring
    is_monitoring = False
    return {'status': 'stopped', 'message': 'voice monitoring stopped'}

@app.route('/emergency-call', methods=['POST'])
def emergency_call():
    '''manual emergency called'''
    print(" Manual emerrgency call triggered")
    return {'status': 'emergency_called', 'message':'Emergency service called'}

@app.route('/status')
def get_status():
    return jsonify({
        'monitorng': is_monitoring,
        'keywords': emergency_keywords,
        'timestamp': datetime.now().isoformat()
    })

'''
running an app if an only if file name is equal to app.py .
this app is running on local host = 127.0.0.1:5050
debug = True is written because it will make changes directly in the website if we save. 
if don't use debug=True then i have to stop running port and again hosting. 
'''
if __name__=="__main__":
    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
    except Exception as e:
        print(f"Microphone error: {e}")
    
    app.run(debug=True, host="0.0.0.0",  port=5050)