
from pathlib import Path
from collections import Counter

# Dataset location
dataset = Path(r"C:\Users\parth\Downloads\LA\LA")

# Label file
label_file = (
    dataset
    / "ASVspoof2019_LA_cm_protocols"
    / "ASVspoof2019.LA.cm.train.trn.txt"
)

if not label_file.exists():
    print("ERROR: Label file not found!")
    print("Check the dataset path.")
    raise SystemExit

counts = Counter()
missing = 0
total = 0

with label_file.open("r", encoding="utf-8") as file:
    for line in file:
        parts = line.split()

        if len(parts) < 5:
            continue

        audio_id = parts[1]
        label = parts[-1]

        audio_path = (
            dataset
            / "ASVspoof2019_LA_train"
            / "flac"
            / f"{audio_id}.flac"
        )

        counts[label] += 1
        total += 1

        if not audio_path.exists():
            missing += 1

print("\n--- ASVspoof 2019 Training Dataset ---")
print("Total labelled samples:", total)
print("Genuine samples:", counts["bonafide"])
print("Spoofed samples:", counts["spoof"])
print("Other labels:", {
    k: v for k, v in counts.items()
    if k not in ("bonafide", "spoof")
})
print("Missing audio files:", missing)