# Speech Emotion Recognition (SER) Project

This project is a Speech Emotion Recognition (SER) system that detects emotions from speech using deep learning techniques. It utilizes an LSTM model trained on the TESS dataset and extracts acoustic features such as pitch, jitter, and tone to classify emotions accurately. The system supports both pre-recorded audio files and real-time microphone inputs.

## Features

- Detects emotions such as happiness, sadness, and anger.
- Displays real-time classification results in the console.
- Supports audio file and live microphone input processing.
- Utilizes acoustic features like pitch, jitter, and tone for analysis.

## Requirements

- Python 3.x
- Required libraries:
  - `librosa` (for audio feature extraction)
  - `numpy` (for numerical operations)
  - `tensorflow` (for LSTM model implementation)
  - `pyaudio` (for real-time audio input)

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/ADITYA2124/SPEECH-EMOTION-RECOGNITION.git
    ```

3. **Download the dataset:**

    - Obtain the TESS dataset from Kaggle and place it in the `data/` directory.

## Usage

1. **Run emotion recognition on an audio file or go for real time emotion detection:**

    ```bash
    python stt1.py
    streamlit run "YOUR LOCAL MACHINE ADDRESS\stt1.py"
    ```

## Files

- `stt1.py`: The main script for emotion recognition.
- `SPEECH EMOTION DETECTION TESS.ipynb`: Script for training the LSTM model.

## Images and Workflow Diagram

![Workflow Diagram](https://github.com/user-attachments/assets/59d5bba0-fc8c-484a-9e35-d84780bac3ab)

*SER Workflow Diagram*


![Sample Results](https://github.com/user-attachments/assets/12cc4088-4d1d-4a98-a690-7742aaff4608)

*Sample Result Preview*


## Contributing

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss proposed modifications.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
