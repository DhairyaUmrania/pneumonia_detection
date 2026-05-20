# RSNA Pneumonia Detection

Object detection and localization of pneumonia opacity regions in chest X-rays using the RSNA Pneumonia Detection Challenge dataset.

This project treats pneumonia detection as a bounding-box localization task rather than a simple image classification problem. The goal is not only to determine whether pneumonia is present, but also to localize the suspicious opacity region in the lung field.

## Project Summary

Pneumonia is a major clinical problem and chest X-rays are one of the most common first-line diagnostic tools. However, pneumonia opacity regions are often diffuse, low contrast, and difficult to localize precisely. This makes the task substantially harder than ordinary image classification.

In this project, we benchmarked several object detection families on the RSNA Pneumonia Detection Challenge dataset:

- Faster R-CNN as a strong two-stage CNN baseline
- YOLO11l as a fast one-stage anchor-free detector
- RT-DETR-l as a modern DETR-style transformer detector
- Weighted Box Fusion ensembles to combine complementary predictions

Our best overall result was achieved using an RT-DETR-l + YOLO11l ensemble.

## Dataset

Dataset: RSNA Pneumonia Detection Challenge from Kaggle

The dataset contains roughly 30,000 frontal chest radiographs with radiologist-drawn bounding boxes around pneumonia opacity regions.

### Splits Used

| Split | Patients | Positive Pneumonia Patients |
|---|---:|---:|
| Train | 21,347 | 4,810 |
| Validation | 2,668 | 622 |
| Test | 2,669 | 580 |

Key dataset characteristics:

- Images are supplied in DICOM format.
- Positive cases may contain one or more bounding boxes.
- Negative images form the majority class.
- Pneumonia opacity boundaries are visually ambiguous.
- Bounding boxes are radiologist annotations, so some inter-annotator variability is expected.

## Task Definition

This is an object detection task.

Classification asks:

```text
Is pneumonia present?
```

Detection asks:

```text
Where is the pneumonia opacity?
```

A prediction is counted as correct only if the predicted bounding box overlaps sufficiently with the ground-truth annotation.

## Evaluation Metrics

The primary metric is mAP@0.5.

| Metric | Meaning |
|---|---|
| mAP@0.5 | Average precision using IoU threshold 0.5 |
| mAP@0.5:0.95 | COCO-style stricter metric averaged from IoU 0.5 to 0.95 |
| AP@0.25 | Relaxed localization metric used for approximate opacity localization analysis |
| Precision | Fraction of predicted boxes that are correct |
| Recall | Fraction of ground-truth boxes detected |

mAP values are expected to be lower than classification accuracy because the model must be correct about both disease presence and spatial localization.

## Preprocessing Pipeline

### 1. DICOM to PNG Conversion

Raw DICOM chest X-rays were decoded and exported as PNG images using `pydicom`.

### 2. Resize to 512 x 512

All images were resized to 512 x 512 pixels for consistent model input and manageable GPU memory use.

### 3. Bounding-Box Scaling Correction

This was a critical fix.

The original RSNA annotations are in 1024 x 1024 coordinate space. Since images were resized to 512 x 512, all bounding-box coordinates were scaled by:

```text
512 / 1024 = 0.5
```

This correction was applied to:

```text
x, y, width, height
```

After the fix, all boxes satisfied:

```text
max x2 <= 512
max y2 <= 512
```

Without this correction, models would train on incorrectly placed boxes.

### 4. Format Conversion

Different model families require different annotation formats:

| Model Family | Format Used |
|---|---|
| YOLO | Normalized `(cx, cy, w, h)` |
| Faster R-CNN | COCO-style / absolute box coordinates |
| DETR / RT-DETR | COCO-style / absolute box coordinates |

## Experiments

### Experiment 1: Faster R-CNN Baseline

Faster R-CNN is a two-stage anchor-based detector. A Region Proposal Network first generates candidate regions, and a second head classifies and refines those boxes.

Result:

| Model | mAP@0.5 |
|---|---:|
| Faster R-CNN | 0.346 |

Role in project:

- Strong CNN-based baseline
- Classical two-stage localization approach
- Useful performance floor for later models

### Experiment 2: YOLO11l at 512 and 640

YOLO11l is a single-stage anchor-free detector. It predicts boxes and classes directly in one forward pass.

Results:

| Model | Input Size | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|
| YOLO11l | 512 | 0.333 | 0.136 |
| YOLO11l | 640 | 0.325 | 0.134 |

Interpretation:

- YOLO11l was competitive but did not surpass Faster R-CNN as a single model.
- Increasing input size from 512 to 640 did not improve performance.
- This suggests that the main bottleneck was not image resolution, but the difficulty of opacity localization.
- YOLO remained useful because its speed and different prediction behavior made it valuable for ensembling.

### Experiment 3: RT-DETR-l at 512

RT-DETR-l is a modern Real-Time Detection Transformer. It combines a CNN-style hybrid encoder with a DETR-style decoder, making it easier to train than vanilla DETR.

Result:

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| RT-DETR-l 512 | 0.402 | 0.395 | 0.355 | 0.145 |

Interpretation:

- Best single-model result.
- Outperformed Faster R-CNN and YOLO11l on mAP@0.5.
- Demonstrated that modern DETR-style detectors are practical for this task, unlike vanilla DETR under limited training.

### Experiment 4: RT-DETR-l + YOLO11l Ensemble

Predictions from RT-DETR-l and YOLO11l were combined using Weighted Box Fusion.

Weighted Box Fusion groups overlapping predictions and averages their coordinates using confidence scores as weights. Unlike NMS, it does not simply discard all but the highest-confidence box.

Result:

| Model | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|
| RT-DETR-l + YOLO11l Ensemble | 0.3802 | 0.1382 |

Interpretation:

- Best AP@0.5 among the main experiments.
- Improved coarse localization and recall at IoU 0.5.
- Slightly lower mAP@0.5:0.95 compared with RT-DETR-l alone, suggesting the ensemble improved detection coverage more than strict box alignment.

## Main Results

Ranked by mAP@0.5:

| Rank | Model | mAP@0.5 | mAP@0.5:0.95 | Notes |
|---:|---|---:|---:|---|
| 1 | RT-DETR-l + YOLO11l Ensemble | 0.3802 | 0.1382 | Best overall AP@0.5 |
| 2 | RT-DETR-l 512 | 0.355 | 0.145 | Best single model |
| 3 | Faster R-CNN | 0.346 | ~0.13 | Strong CNN baseline |
| 4 | YOLO11l 512 | 0.333 | 0.136 | Competitive one-stage model |
| 5 | YOLO11l 640 | 0.325 | 0.134 | Higher resolution did not help |

## Additional Ensemble Sweep

We also evaluated combinations of the available trained detectors using Weighted Box Fusion. The best AP@0.5 and AP@0.25 were achieved by combining YOLO11l at 512 and YOLO11l at 640.

| Ensemble | WBF IoU | AP@0.25 | AP@0.5 | AP@0.5:0.95 |
|---|---:|---:|---:|---:|
| YOLO512 + YOLO640 | 0.35 | 0.6158 | 0.4273 | 0.1373 |
| YOLO512 + YOLO640 | 0.55 | 0.5689 | 0.4185 | 0.1504 |
| YOLO512 + YOLO640 | 0.45 | 0.5809 | 0.4176 | 0.1462 |
| RT-DETR + YOLO512 | 0.45 | 0.5635 | 0.4123 | 0.1388 |
| RT-DETR + YOLO512 | 0.35 | 0.5871 | 0.4084 | 0.1262 |
| RT-DETR + YOLO640 | 0.45 | 0.5608 | 0.4077 | 0.1370 |

Best by metric:

| Category | Best Model | Setting | Score |
|---|---|---:|---:|
| Best AP@0.25 | YOLO512 + YOLO640 | WBF IoU 0.35 | 0.6158 |
| Best AP@0.5 | YOLO512 + YOLO640 | WBF IoU 0.35 | 0.4273 |
| Best AP@0.5:0.95 among ensembles | YOLO512 + YOLO640 | WBF IoU 0.55 | 0.1504 |
| Best single model AP@0.5 | RT-DETR-l 512 | Single model | 0.355 |
| Best single model AP@0.5:0.95 | RT-DETR-l 512 | Single model | 0.145 |

## Qualitative Detection Outputs

Visual inspection was used to evaluate whether model predictions were anatomically meaningful.

Visualization convention:

- Green boxes: Ground-truth RSNA annotations
- Red boxes: Model predictions

The qualitative outputs showed that predictions often landed in the same opacity region as the ground truth, even when box size or exact boundary differed. This is expected because pneumonia opacity does not have sharp object-like edges.

The model sometimes produced multiple candidate boxes around the same opacity region. This reflects uncertainty in the extent of the abnormal region and the behavior of ensemble fusion.

## Comparison with Published Results

Prior work on the RSNA Pneumonia Detection Challenge reports similar-scale mAP values when using challenge-style detection evaluation.

| Study / Source | Method | Reported Result |
|---|---|---:|
| RSNA challenge leaderboard historical best | Competition submissions | mAP ~ 0.2547 |
| Rajaraman et al. | RetinaNet ensemble with CXR-specific initialization | mAP 0.3272 |
| DeepRadiology Team | CoupleNet-style detector with ensemble | Ensemble score 0.2310 |
| Wu et al. | Anchor-free detector with feature pyramid | mAP 0.3120 |
| This project | RT-DETR-l + YOLO11l ensemble | AP@0.5 0.3802 |

Important note: these numbers are contextual benchmarks, not strict head-to-head comparisons. Different papers may use different splits, preprocessing pipelines, metric implementations, and confidence thresholds.

## Key Findings

1. RT-DETR-l was the strongest single model.
2. Vanilla DETR failed to converge under limited training, while RT-DETR-l worked effectively.
3. YOLO11l was competitive, but increasing resolution from 512 to 640 did not improve performance.
4. Weighted Box Fusion improved AP@0.5 by combining complementary detector predictions.
5. Relaxed localization metrics such as AP@0.25 are useful for pneumonia because opacity boundaries are diffuse and subjective.
6. Results are directionally competitive with published RSNA benchmarks.

## Future Directions

### Domain-Adaptive Pretraining

Initialize model backbones using large chest X-ray datasets such as CheXpert or NIH ChestXray14 instead of ImageNet.

### Class-Aware Augmentation

Apply heavier augmentation to positive pneumonia cases to mitigate class imbalance without discarding negative examples.

### Multi-Scale DETR Variants

Explore Deformable DETR or Anchor DETR for better multi-resolution feature fusion and small-opacity detection.

### Confidence Calibration

Use Platt scaling, temperature scaling, or similar techniques to improve the reliability of model confidence scores.

### Human-AI Reader Study

Compare model localization performance against radiologists on a blinded held-out set, and test whether AI assistance improves radiologist performance.

### Semi-Supervised Learning

Use pseudo-labeling or consistency regularization on unlabeled chest X-rays to increase effective training data.

## Repository Structure

```text
pneumonia_detection/
├── configs/
│   └── config.py
├── data/
│   ├── dataset.py
│   ├── augmentations.py
│   └── download.py
├── models/
│   ├── faster_rcnn.py
│   ├── detr_baseline.py
│   ├── detr_multiscale.py
│   └── domain_pretrain.py
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_all_experiments.sh
├── ensemble_eval.py
├── visualize_boxes.py
├── prepare_yolo_full.py
├── prepare_yolo_sample.py
├── prepare_yolo_rsna.py
├── requirements.txt
├── README.md
└── HOW_TO_RUN.md
```

## Setup

Create and activate a Python environment:

```bash
conda create -n pneumonia python=3.10 -y
conda activate pneumonia
pip install -r requirements.txt
```

## Download Data

The RSNA dataset is not included in this repository due to size and licensing restrictions.

Download it from Kaggle after accepting the competition rules:

```bash
python data/download.py --data-dir ./rsna_data
```

## Preprocess Data

```bash
python scripts/preprocess.py \
  --input-dir ./rsna_data \
  --output-dir ./rsna_processed \
  --target-size 512
```

## Prepare YOLO Dataset

```bash
python prepare_yolo_full.py
```

This creates a YOLO-compatible directory and `rsna.yaml` file.

## Train YOLO11l

Example:

```bash
yolo detect train \
  model=yolo11l.pt \
  data=rsna_yolo/rsna.yaml \
  imgsz=512 \
  epochs=100 \
  batch=8 \
  device=0 \
  workers=2 \
  plots=False \
  project=outputs_yolo \
  name=yolo11l_rsna_512
```

## Train RT-DETR-l

Example:

```bash
yolo detect train \
  model=rtdetr-l.pt \
  data=rsna_yolo/rsna.yaml \
  imgsz=512 \
  epochs=80 \
  batch=4 \
  device=0 \
  workers=2 \
  plots=False \
  project=outputs_rtdetr \
  name=rtdetr_l_rsna_512
```

## Run Ensemble Evaluation

Example:

```bash
python ensemble_eval.py \
  --data rsna_yolo/rsna.yaml \
  --imgsz 512 \
  --conf 0.001 \
  --wbf-iou 0.55 \
  --models \
    runs/detect/outputs_rtdetr/rtdetr_l_rsna_512_bs4_a6000_gpu2/weights/best.pt \
    runs/detect/outputs_yolo/yolo11l_rsna_512_bs16_gpu2/weights/best.pt
```

## Visualize Predictions

Example:

```bash
python visualize_boxes.py \
  --data rsna_yolo/rsna.yaml \
  --imgsz 512 \
  --conf 0.001 \
  --wbf-iou 0.35 \
  --positive-only \
  --num-images 40 \
  --out-dir visualizations_best_pred_per_gt \
  --models \
    runs/detect/outputs_yolo/yolo11l_rsna_512_bs16_gpu2/weights/best.pt \
    runs/detect/outputs_yolo/yolo11l_rsna_640_bs4_a6000_gpu2-2/weights/best.pt
```

## Notes on Large Files

The following are intentionally excluded from Git:

- Raw RSNA data
- Preprocessed images
- YOLO dataset folders
- Model checkpoints
- Trained weights
- Prediction visualizations
- Experiment logs

See `.gitignore` for details.

## Team

- Dhairya Umrania
- Rishabh Gosain
- Devaansh Kataria

## Acknowledgments

This project was completed for AMS 563: Medical Image Analysis.

The RSNA Pneumonia Detection Challenge dataset is available through Kaggle after accepting the competition rules.
