from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(".").resolve()
IMG_SRC = ROOT / "rsna_processed" / "images"
OUT = ROOT / "rsna_yolo"
IMG_SIZE = 512

SPLITS = {
    "train": ROOT / "rsna_processed" / "train_split.csv",
    "val": ROOT / "rsna_processed" / "val_split.csv",
    "test": ROOT / "rsna_processed" / "test_split.csv",
}

if OUT.exists():
    shutil.rmtree(OUT)

for split in SPLITS:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

def write_split(split, csv_path):
    df = pd.read_csv(csv_path)
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
        pos = g[g["Target"] == 1]

        for _, row in pos.iterrows():
            x = float(row["x"])
            y = float(row["y"])
            w = float(row["w"])
            h = float(row["h"])

            x_center = (x + w / 2.0) / IMG_SIZE
            y_center = (y + h / 2.0) / IMG_SIZE
            width = w / IMG_SIZE
            height = h / IMG_SIZE

            vals = [x_center, y_center, width, height]
            vals = [min(max(v, 0.0), 1.0) for v in vals]

            if vals[2] > 0 and vals[3] > 0:
                lines.append(f"0 {vals[0]:.6f} {vals[1]:.6f} {vals[2]:.6f} {vals[3]:.6f}")

        dst_label.write_text("\n".join(lines) + ("\n" if lines else ""))

        copied += 1
        if lines:
            positives += 1

    print(f"{split}: copied={copied}, positive_images={positives}, missing={missing}")

for split, csv_path in SPLITS.items():
    write_split(split, csv_path)

yaml_text = f"""path: {OUT}
train: images/train
val: images/val
test: images/test

names:
  0: pneumonia
"""

(OUT / "rsna.yaml").write_text(yaml_text)
print("\nWrote:", OUT / "rsna.yaml")
