from pathlib import Path
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(".").resolve()
IMG_SRC = ROOT / "rsna_processed" / "images"
OUT = ROOT / "rsna_yolo"

IMG_SIZE = 512

splits = {
    "train": ROOT / "rsna_processed" / "train_split.csv",
    "val": ROOT / "rsna_processed" / "val_split.csv",
    "test": ROOT / "rsna_processed" / "test_split.csv",
}

for split in splits:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

def write_split(split, csv_path):
    df = pd.read_csv(csv_path)

    # Group rows by patientId because one image can have multiple boxes
    grouped = df.groupby("patientId", sort=False)

    copied = 0
    positives = 0
    missing = 0

    for pid, g in grouped:
        src_img = IMG_SRC / f"{pid}.png"
        if not src_img.exists():
            missing += 1
            continue

        dst_img = OUT / "images" / split / f"{pid}.png"
        dst_label = OUT / "labels" / split / f"{pid}.txt"

        shutil.copy2(src_img, dst_img)

        lines = []
        pos = g[g["Target"] == 1].copy()

        for _, row in pos.iterrows():
            x = float(row["x"])
            y = float(row["y"])
            w = float(row["w"])
            h = float(row["h"])

            # Convert top-left xywh pixels to YOLO normalized center xywh
            x_center = (x + w / 2.0) / IMG_SIZE
            y_center = (y + h / 2.0) / IMG_SIZE
            width = w / IMG_SIZE
            height = h / IMG_SIZE

            # Clip just in case of tiny floating point boundary issues
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            width = min(max(width, 0.0), 1.0)
            height = min(max(height, 0.0), 1.0)

            if width > 0 and height > 0:
                lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # For negative images, YOLO accepts an empty label file
        dst_label.write_text("\n".join(lines) + ("\n" if lines else ""))

        copied += 1
        if lines:
            positives += 1

    print(f"{split}: copied={copied}, positive_images={positives}, missing_images={missing}")

for split, path in splits.items():
    write_split(split, path)

yaml_text = f"""path: {OUT}
train: images/train
val: images/val
test: images/test

names:
  0: pneumonia
"""

(OUT / "rsna.yaml").write_text(yaml_text)
print("\nWrote:", OUT / "rsna.yaml")
