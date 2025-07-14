# Sano Awaj
Sano Awaj is a web app that helps woman and underprivileged users report and prevent harrasment in a safe and easy way. It gives users tools to protect themselves, voice activated emergency call to police or trusted one and recording voice when trigger word or dangerous voice tone is detected. 

# Is this a necessary?
- yes, Many people in Nepal, especially women and marginalized groups, suffer silently because there’s no safe way to speak out.

- Public buses, streets, and even homes are often unsafe.

- This app gives people a voice, even when they feel powerless.

- By combining real-time safety tools and community reporting, we can build a smarter, safer Kathmandu.

# Tech stack Used By me
## Machine learning 
- step1: Dataset Prepration and Sources 
  - kaggle CREMA - D
  - RAVDESS Emotional speech
  - Surrey Audio - visual Expressions
  - Toronto emotional speech 
  - common voice nepal
  - all .wav files recommended
  - still need for the screaming, crying and asking help (bachau) by Nepali Women
- step2: Model Training
  - using tensorflow/keras RNN or LSTM 
  - using librosa library
  - save model in h5
  - Django Backend Setup
    - /predict-audio/	POST |	Accept audio clip | run model prediction every 2 sec
    - /trigger-alert/	POST | trigger recording video | calling if danger
- step 3: connecting with Frontend 
  - Audio sent every 2-3 seconds via fetch()
  - if danger becomes true and,
    - starts video recording through WebRTC
    - or call user trusted one or poliice
- step 4: deployment
  - Django Backend in (Render / railway)
  - frontend react in vercel/ netlify
  - model in static directory 

These are stages that need to completed by me aka shiva chaudhary 

# Team members
- Shiva chaudhary
- Chandan Sharma Thakur 
- Umanga ghimire 