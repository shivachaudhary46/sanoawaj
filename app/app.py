'''
import all necessary libraries flask, 
class including jsonify, request, render_template, redirect, url_for
import speech_recognition for audio preprocessing and taaking input
import time to create a datetime stamp value 
'''
from flask import Flask, jsonify, request, render_template, redirect, url_for
import time
import speech_recognition as sr
import queue 
import json 
import threading
from datetime import datetime
import pyttsx3

'''
intialising Flask app name 
'''
app = Flask(__name__)

''' variables '''
voice = []
is_monitoring = True
microphone = sr.Microphone()
recognizer = sr.Recognizer()

emergency_keywords = [
    'help', 'emergency', 'police', 'danger', 'scared', 'attack',
    'save me', 'help me', 'call police', 'stranger', 'following'
]

'''function to convert text to speech'''
def SpeakText(command):

    # initalize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAwait()

'''
I will make ismonitoring a variable flag for iterating in a loop 
if ismonitoring flag becomes true when users click start monitoring 
and my app will try to listen. 
'''
def listen_continuously():
    '''Background voice listening function'''
    global is_monitoring
    
    # wait for a second to let the recognizer
    # adjust the energy threshold based on 
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

    # this loop will only execute if user clicks start audio listening. 
    # i can manage by adding addEventListener
    while is_monitoring:
        try:
            # listen for audio with timeout
            with microphone as source:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=5)

            # put audio in queue with timeout
            voice.append(audio)

            # using google to recognize audio
            text = recognizer.recognize_google(audio).lower()
            print(text)
        # if no audio is detected in total of 3 sec 
        # continue listening voice 
        except sr.WaitTimeoutError:
            pass
            
        # if any error occurs  
        except Exception as e: 
            print(f"listening error: {e}")

        # wait 2 seconds before listening to next audio
        time.sleep(4)

'''
process voice audio from queue
'''
def process_voice():

    global is_monitoring

    while is_monitoring:
        try:
            if not voice.empty(): 
                # extract the voice from queue
                audio = voice.pop(0)
                print(audio)

                # convert speech to text
                try:
                    text = recognizer.recognize_google(audio).lower()
                    print("heard:", text)

                    # we can also add emergency keywords and check if it 
                    # is present or not
                    if any(keyword in text for keyword in emergency_keywords):
                        print(f"emergency detected: {text}")
                    
                        # i will add my Model function

                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print(f"speech recognition error: {e}")

        except queue.Empty:
            pass # could not understand audio
        except Exception as e:
            print(f"error processing audio: {e}")

def trigger_emergency_logging(detected_text):
    """Trigger emergency response when keyword detected"""
    timestamp = datetime.now().isoformat()

    '''create emergency json file'''
    emergency = {
        'timestamp': timestamp,
        'detected_text': 'detected_text',
        'status': 'emergency_detected'
    }

    '''append jsonify  in json file'''
    with open('voice_alerts.json', 'a') as f:
        data = json.dumps(emergency)
        f.write(data)

    print("emergency response triggered!")

    '''voice model prediction function'''

@app.route('/', methods=['POST', 'GET'])
def home():
    return render_template("home.html")

@app.route('/start-monitoring', methods=['POST'])
def start_monitoring():
    global is_monitoring

    if not is_monitoring:
        is_monitoring = True

        # start background threads 
        listen_thread = threading.Thread(target=listen_continuously, daemon=True)
        process_thread = threading.Thread(target=process_voice, daemon=True)

        listen_thread.start()
        process_thread.start()

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