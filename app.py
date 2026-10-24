
import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import io
from pathlib import Path

st.set_page_config(
    page_title="AI Fake Voice Detector",
    page_icon="🎙️",
    layout="centered"
)

SAMPLE_RATE = 16000
DURATION = 3
N_MELS = 64
MODEL_PATH = Path(__file__).parent / "voice_cnn.pth"


# CNN architecture must match the trained model
class VoiceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@st.cache_resource
def load_model():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )
    model = VoiceCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def extract_feature(audio_bytes):
    audio, _ = librosa.load(
        io.BytesIO(audio_bytes),
        sr=SAMPLE_RATE,
        mono=True,
        duration=DURATION
    )

    if len(audio) == 0:
        raise ValueError("The uploaded audio is empty.")

    target_length = SAMPLE_RATE * DURATION

    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=1024,
        hop_length=256,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    return audio, mel_db.astype(np.float32)


st.title("🎙️ AI-Powered Fake Voice Detector")
st.write(
    "Upload a voice recording to check whether it resembles "
    "genuine or spoofed speech from the model's training data."
)

st.info(
    "Prototype note: this model was trained on a small subset of "
    "ASVspoof 2019 LA. Results may be unreliable for modern AI voices "
    "or recordings unlike its training data."
)

audio_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "flac", "ogg"]
)

if audio_file is not None:
    st.audio(audio_file)

    if st.button("🔍 Analyze Voice", type="primary"):
        try:
            if not MODEL_PATH.exists():
                st.error("Model file voice_cnn.pth was not found.")
                st.stop()

            with st.spinner("Analyzing audio..."):
                model = load_model()
                audio, mel_db = extract_feature(audio_file.getvalue())

                tensor = torch.from_numpy(mel_db).unsqueeze(0).unsqueeze(0)

                with torch.no_grad():
                    logits = model(tensor)
                    probabilities = torch.softmax(logits, dim=1)[0]

                human_prob = probabilities[0].item()
                spoof_prob = probabilities[1].item()
                predicted_class = int(torch.argmax(probabilities).item())

            st.subheader("Analysis Result")

            if predicted_class == 0:
                st.success("Prediction: Human / Bona fide")
                confidence = human_prob
            else:
                st.warning("Prediction: Spoof-like / Potentially fake")
                confidence = spoof_prob

            st.metric("Model confidence", f"{confidence * 100:.1f}%")
            st.progress(confidence)

            st.caption(
                "Confidence is the model's output score, not a guarantee "
                "that the recording is genuine or fake."
            )

            st.write("### Class scores")
            st.write(f"Human / Bona fide: {human_prob * 100:.1f}%")
            st.write(f"Spoof: {spoof_prob * 100:.1f}%")

            st.write("### Audio Details")
            st.write(f"Duration analyzed: {len(audio) / SAMPLE_RATE:.2f} seconds")
            st.write(f"Sample rate: {SAMPLE_RATE} Hz")
            st.write("Channels analyzed: Mono")
            st.write("Feature extracted: 64-band Mel-spectrogram")

            fig, ax = plt.subplots()
            image = librosa.display.specshow(
                mel_db,
                sr=SAMPLE_RATE,
                hop_length=256,
                x_axis="time",
                y_axis="mel",
                ax=ax
            )
            fig.colorbar(image, ax=ax, format="%+2.0f dB")
            ax.set_title("Mel-Spectrogram")
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"Analysis failed: {e}")