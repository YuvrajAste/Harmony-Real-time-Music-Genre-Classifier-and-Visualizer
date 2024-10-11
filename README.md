# 🎶 Music Genre Classification with Real-Time Audio Visualization 🎶

This project uses machine learning to classify music genres based on audio files. Users can upload audio files through a web interface, visualize the waveform in real-time, and get predictions on the music genre. The model is built using audio features such as MFCCs, chroma, mel-spectrogram, and spectral contrast.

## Features
- 🎵 **Upload Audio Files**: Users can upload `.wav` audio files for classification.
- 🎧 **Real-Time Audio Playback & Visualization**: Audio waveform is visualized using the Web Audio API.
- 🎤 **Music Genre Classification**: The backend model predicts the genre from audio features extracted using **librosa**.
- 🔗 **Informative Links**: The app provides a link to Britannica for more information on the predicted genre.

## Tech Stack
- **Frontend**: HTML, JavaScript (for real-time visualization)
- **Backend**: Python, Flask
- **Machine Learning**: SVM classifier, **librosa** for feature extraction
- **Model**: Trained on a dataset of 7 genres (blues, heavy metal, jazz, hip-hop, reggae, country, disco)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/music-genre-classification.git
cd music-genre-classification
```
### 2. Set up a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # For Linux/MacOS
venv\Scripts\activate  # For Windows
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the Flask app
```bash
python app.py
```

### Usage
Upload an audio file in .wav format.
Play the audio file and watch the real-time visualization.
Get the predicted genre and explore related content on Britannica.

### Dataset
The model is trained on a subset of the [GTZAN dataset](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)
. Audio features like MFCCs, chroma, mel-spectrogram, and spectral contrast are extracted from each file.

### Model Accuracy
The model was fine-tuned using GridSearchCV and has an accuracy of **72.8%** on the test set.

### Visualizations
The project also includes Principal Component Analysis (PCA) visualizations of the music genre features, allowing us to understand how different genres are clustered in a reduced 2D space.

### Screenshots
![image](https://github.com/user-attachments/assets/e65566f4-5fb0-4643-a842-3f675369070e)

### Audio File Upload
![image](https://github.com/user-attachments/assets/b2361c07-7d0d-406c-8cff-c6744553e9d1)

### Preidicting the Genre and Visulaizing Audio.
![image](https://github.com/user-attachments/assets/533b844c-5673-493a-b097-c7cccac04dd8)

Future Improvements
Integrate support for more audio formats (e.g., MP3).
Improve real-time audio feature extraction and visualization.
Explore deep learning models like CNNs for more accurate predictions.
License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT)
 - see the LICENSE file for details.
