
from pathlib import Path
import random
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Dataset paths
ROOT = Path(r"C:\Users\parth\Downloads\LA\LA")
PROTOCOL = ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
AUDIO = ROOT / "ASVspoof2019_LA_train" / "flac"

# Settings for a small first experiment
SAMPLE_RATE = 16000
DURATION = 3
N_MELS = 64
LIMIT_PER_CLASS = 500
BATCH_SIZE = 16
EPOCHS = 5

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Read labelled audio paths
samples = {"bonafide": [], "spoof": []}

with PROTOCOL.open("r", encoding="utf-8") as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 5:
            audio_id = parts[1]
            label = parts[-1]
            path = AUDIO / f"{audio_id}.flac"

            if label in samples and path.exists():
                samples[label].append(path)

# Balance the initial experiment
for label in samples:
    random.shuffle(samples[label])
    samples[label] = samples[label][:LIMIT_PER_CLASS]

paths = samples["bonafide"] + samples["spoof"]
labels = [0] * len(samples["bonafide"]) + [1] * len(samples["spoof"])

if len(set(labels)) < 2:
    raise RuntimeError("Could not find both genuine and spoof samples.")

print("Genuine:", len(samples["bonafide"]))
print("Spoof:", len(samples["spoof"]))
print("Preparing Mel-spectrograms...")

def extract_feature(path):
    audio, _ = librosa.load(
        path, sr=SAMPLE_RATE, mono=True, duration=DURATION
    )

    # Fixed length: 3 seconds
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

    # Normalize each spectrogram
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db.astype(np.float32)

features = []
valid_labels = []

for i, (path, label) in enumerate(zip(paths, labels)):
    try:
        features.append(extract_feature(path))
        valid_labels.append(label)
    except Exception as e:
        print("Skipping:", path.name, str(e))

    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1}/{len(paths)} files")

X = np.array(features, dtype=np.float32)
y = np.array(valid_labels, dtype=np.int64)

if len(y) < 20 or len(np.unique(y)) < 2:
    raise RuntimeError("Not enough valid samples to train.")

# Split after feature extraction
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).unsqueeze(1)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(
    AudioDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_loader = DataLoader(
    AudioDataset(X_val, y_val),
    batch_size=BATCH_SIZE
)

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

model = VoiceCNN().to(device)

# Balanced class weights for the loss function
class_counts = np.bincount(y_train, minlength=2)
weights = len(y_train) / (2.0 * np.maximum(class_counts, 1))
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(weights, dtype=torch.float32, device=device)
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    predictions, actual = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb.to(device))
            predictions.extend(logits.argmax(1).cpu().numpy())
            actual.extend(yb.numpy())

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {total_loss / max(len(train_loader), 1):.4f}"
    )

print("\nValidation results:")
print(classification_report(
    actual, predictions,
    labels=[0, 1],
    target_names=["Human", "AI/Spoof"],
    zero_division=0
))

# Save the trained model and configuration
torch.save({
    "model_state_dict": model.cpu().state_dict(),
    "n_mels": N_MELS,
    "sample_rate": SAMPLE_RATE,
    "duration": DURATION,
    "labels": {"human": 0, "spoof": 1}
}, "voice_cnn.pth")

print("\nModel saved as voice_cnn.pth")