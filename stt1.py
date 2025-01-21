import os
import sounddevice as sd
import numpy as np
import librosa
import librosa.display
import streamlit as st
from keras.models import load_model
from scipy.io.wavfile import write
import tempfile
import matplotlib.pyplot as plt

# Load the pre-trained model
model_path = 'D:/PYTHON PROGRAMS/ser_project/ERM.h5'
model = load_model(model_path)

# Emotion-to-emoji mapping
emotion_to_emoji = {
    "HAPPY": "D:/PYTHON PROGRAMS/ser_project/happy.png",
    "SAD": "D:/PYTHON PROGRAMS/ser_project/sad.png",
    "ANGRY": "D:/PYTHON PROGRAMS/ser_project/angry.png",
    "SURPRISED": "D:/PYTHON PROGRAMS/ser_project/surprised.png",
    "NEUTRAL": "D:/PYTHON PROGRAMS/ser_project/neutral.png",
    "FEAR": "D:/PYTHON PROGRAMS/ser_project/fear.png",
    "DISGUST": "D:/PYTHON PROGRAMS/ser_project/disgust.png"
}

label_mapping = {0: 'ANGRY', 1: 'DISGUST', 2: 'FEAR', 3: 'HAPPY', 4: 'NEUTRAL', 5: 'SURPRISED', 6: 'SAD', 
                7: 'ANGRY', 8: 'DISGUST', 9: 'FEAR', 10: 'HAPPY', 11: 'NEUTRAL', 12: 'SURPRISED', 13: 'SAD'}

# Function to extract features from the audio
def extract_features(audio_data):
    audio, sr = librosa.load(audio_data, res_type='scipy')
     # Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=20)
    audio = librosa.util.normalize(audio)
    # Plot waveplot
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr, ax=ax, color='blue')
    ax.set_title("Waveplot")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    n_fft_value = min(1024, len(audio))
    # Compute features
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft_value).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft_value).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=n_fft_value).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)

    
    # Concatenate features
    features = np.concatenate([mfcc, chroma, mel, spectral_contrast, tonnetz, rms, zcr], axis=0)
    
    return np.expand_dims(np.expand_dims(features, axis=0), axis=0)

# Streamlit App
st.title("Emotion Prediction App")

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Choose an Option", ["Home", "Record Audio", "Audio File", "Prediction History"])

# Global history
if "history" not in st.session_state:
    st.session_state["history"] = []

# Home Page
if choice == "Home":
    st.write("## Welcome to the Emotion Prediction App!")
    st.write("""
        This app predicts emotions from speech files or via your microphone. 
        Simply upload a `.wav` file or record audio using the microphone.
    """)

# Record Audio Page
elif choice == "Record Audio":
    st.write("## Record Audio Using Microphone")

    # Record button
    if st.button("Start Recording"):
        fs = 16000  # Sampling frequency
        duration = 5  # Duration in seconds

        st.write("Recording...")

        # Record the audio
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()

        # Save the recording to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            write(temp_file.name, fs, recording)
            temp_file_path = temp_file.name
        st.audio(temp_file_path, format="audio/wav")
        st.write("Recording complete!")

        # Predict emotion from the recorded audio
        features = extract_features(temp_file_path)
        try:
            predicted_probabilities = model.predict(features)
            #st.write("Predicted probabilities:", predicted_probabilities)
            predicted_label_index = np.argmax(predicted_probabilities)
            predicted_emotion = label_mapping[predicted_label_index]
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            st.stop()

        # Display the result
        st.write(f"### Predicted Emotion: {predicted_emotion}")
        emoji_path = emotion_to_emoji.get(predicted_emotion)
        if emoji_path:
            st.image(emoji_path, width=100)

        # Save the prediction to history
        st.session_state["history"].append((temp_file_path, predicted_emotion))

# Audio File Page
elif choice == "Audio File":
    st.write("## Upload an Audio File")
    uploaded_file = st.file_uploader("Choose a `.wav` file", type=["wav"])

    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.audio(temp_file_path, format="audio/wav")
        # Predict emotion
        try:
            features = extract_features(temp_file_path)
            predicted_probabilities = model.predict(features)
            #st.write("Predicted probabilities:", predicted_probabilities)
            predicted_label_index = np.argmax(predicted_probabilities)
            predicted_emotion = label_mapping[predicted_label_index]
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            st.stop()

        # Update history
        st.session_state["history"].append((uploaded_file.name, predicted_emotion))

        # Display emotion and emoji
        st.write(f"### Predicted Emotion: {predicted_emotion}")
        emoji_path = emotion_to_emoji.get(predicted_emotion)
        if emoji_path:
            st.image(emoji_path, width=100)

        # Clean up temp file
        os.remove(temp_file_path)

# Prediction History Page
elif choice == "Prediction History":
    st.write("## Prediction History")
    if st.session_state["history"]:
        for idx, (file_name, emotion) in enumerate(st.session_state["history"], start=1):
            st.write(f"{idx}. File: **{file_name}** | Emotion: **{emotion}**")
    else:
        st.write("No prediction history available.")
