# SanoAwaj
Sano Awaj is our hackathon project. It is a web app that helps woman and underprivileged users report and prevent harrasment in a safe and easy way.

# Plan of Attack for team member (shiva chaudhary)
### step 1: preparing dataset from kaggle
***NOTE :*** This dataset will be used for the speech and emotion recognition. I need more dataset for stress signals like (bachau, help, screaming by ***NEPALI*** women)
- CREMA-D
- RAVDESS Emotional speech audio
- Surrey Audio-Visual Expressed Emotion (SAVEE)
- Toronto emotional speech set (TESS)
- Common Voice (Nepali) 
- OpenSLR SLR64

### step 2: Preprocessing Code 
***NOTE :*** we will be using librosa, tensorflow, keras for audio preprocessing and model training 
- CNN model will be used for image recognition for finding pattern. Eventhough, I'm working with audio, I need to convert audio into visual features like MFCCs(Mel Frequency Cepstral Coefficients) or spectogram and these are just 2D images
- If I want to detect pattern overtime and full sentences or emotion tone oveer 5 - 10 secs

### step 3: Django as Backend Setup
***NOTE :*** I don't know Django. I need to learn in just ***2 Days***
- /predict-audio/ using POST method. This function accept audio clip, run model prediction
- /trigger-alert/ using POST method. This function trigger a call to police and record a video 

### step 4: Connecting with Frontend
- Audio will be sent every 2-3 seconds via fetch()
- if model predict 'danger' then 
  - Starts video recording (WebRTC)
  - Calls to most trusted one looking phone contacts or police

### step 5: Deployment with SEO
***NOTE :***so, This is not my main point i 'll try to help Chandan in this. 

### sample Workflow loop (live flow)
- Mic records audio on frontend via JS every 2 sec
- fetch("/predict-audio/") sends audio to Django
- Django runs .h5 model stored and returns prediction on real time.
- If "danger" detected:
  - Sends call via Twilio API
  - Triggers or alert on frontend

### Team members 
` 
- Shivshakti Chaudhary
- Chandan Sharma Thakur 
- Umanga Ghimire
`