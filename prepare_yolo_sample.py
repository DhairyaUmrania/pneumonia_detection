from pathlib import Path
import shutil
import pandas as pd
import random

ROOT = Path(".").resolve()
IMG_SRC = ROOT / "rsna_processed" / "images"
OUT = ROOT / "rsna_yolo_sample"
IMG_SIZE = 512
random.seed(42)

SPLIT_CSVS = {
    "train": ROOT / "rsna_processed" / "train_split.csv",
    "val": ROOT / "rsna_processed" / "val_split.csv",
}

# Number of images for quick test
N = {
    "train": {"pos": 50, "neg": 50},
    "val": {"pos": 20, "neg": 20},
}

def reset_dirs():
    if OUT.exists():
        shutil.rmtree(OUT)

    for split in ["train", "val"]:
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

def select_patient_ids(df, n_pos, n_neg):
    patient_targets = df.groupby("patientId")["Target"].max()

    pos_ids = patient_targets[patient_targets == 1].index.tolist()
    neg_ids = patient_targets[patient_targets == 0].index.tolist()

    random.shuffle(pos_ids)
    random.shuffle(neg_ids)

    return pos_ids[:n_pos] + neg_ids[:n_neg]

def write_yolo_split(split, csv_path, n_pos, n_neg):
    df = pd.read_csv(csv_path)
    selected_ids = select_patient_ids(df, n_pos, n_neg)

    copied = 0
    positives = 0
    missing = 0

    for pid in selected_ids:
        img_path = IMG_SRC / f"{pid}.png"
        if not img_path.exists():
            missing += 1
            continue

        g = df[df["patientId"] == pid]

        dst_img = OUT / "images" / split / f"{pid}.png"
        dst_label = OUT / "labels" / split / f"{pid}.txt"

        shutil.copy2(img_path, dst_img)

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

    print(f"{split}: copied={copied}, positives={positives}, missing={missing}")

reset_dirs()

for split, csv_path in SPLIT_CSVS.items():
    write_yolo_split(
        split,
        csv_path,
        N[split]["pos"],
        N[split]["neg"],
    )

yaml_text = f"""path: {OUT}
train: images/train
val: images/val

names:
  0: pneumonia
"""

(OUT / "rsna_sample.yaml").write_text(yaml_text)
print("\nWrote:", OUT / "rsna_sample.yaml")
