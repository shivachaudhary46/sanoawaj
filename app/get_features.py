import numpy as np
import pandas as pd
import librosa

def stretch(data, rate=0.8):
    # stretching time speeding up/slowing down by  rate 0.8
    return librosa.effects.time_stretch(data, rate=rate)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=-6)

def noise(data):
    # uniform would generate a random integer and we will multiply by a maximum value in a given array
    adding_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + adding_amp * np.random.normal(size=data.shape[0])
    return data

def extract_features(data, sample_rate):
    result = np.array([])

    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    # RMS
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    # MelSpectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result

def get_features(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2))

    # data with stretching and pitching 
    stretch_data = stretch(data)
    pitch_data = pitch(stretch_data, sample_rate)
    res4 = extract_features(pitch_data, sample_rate)
    result = np.vstack((result, res4))

    return result

def invert_prediction(model, X_test, y_test, encoder): 
    pred_test = model.predict(X_test)
    y_pred = encoder.inverse_transform(pred_test)
    y_test_inv = encoder.inverse_transform(y_test)
    
    return y_pred, y_test_inv