from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import librosa
import numpy as np
import joblib
import os

# Initialize Flask App
app = Flask(__name__)

# Folder to store uploaded audio files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model = joblib.load('best_genre_classifier.pkl')

# Dictionary mapping genres to Britannica URLs
genre_urls = {
    'blues': 'https://www.britannica.com/art/blues-music',
    'heavy metal': 'https://www.britannica.com/art/heavy-metal-music',
    'jazz': 'https://www.britannica.com/art/jazz',
    'hip-hop': 'https://www.britannica.com/art/hip-hop',
    'reggae': 'https://www.britannica.com/art/reggae',
    'country': 'https://www.britannica.com/art/country-music',
    'disco': 'https://www.britannica.com/art/disco'
}

# Function to extract features from the audio file
def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    spectral_contrast = librosa.feature.spectral_contrast(S=np.abs(librosa.stft(audio)), sr=sample_rate)

    features = np.hstack([
        np.mean(mfccs.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(mel.T, axis=0),
        np.mean(spectral_contrast.T, axis=0)
    ])
    return features

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to serve the uploaded audio files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path) 

        # Extract features and make a prediction
        features = extract_features(file_path)
        features = np.array([features])
        genre_prediction = model.predict(features)[0]
        
        # Create a Britannica URL for the predicted genre
        genre_key = genre_prediction.lower()
        britannica_url = genre_urls.get(genre_key, "#")  # Get the URL or default to "#"
        
        # Pass the file URL to play the audio
        file_url = url_for('uploaded_file', filename=file.filename)
        
        return render_template('index.html', 
                               prediction=genre_prediction,
                               britannica_url=britannica_url,
                               file_url=file_url)

# Main function to run the Flask app
if __name__ == '__main__':
    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
