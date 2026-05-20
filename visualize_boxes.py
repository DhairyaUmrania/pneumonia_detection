import argparse
import yaml
from pathlib import Path
import random
import numpy as np
import cv2
from ultralytics import YOLO
from tqdm import tqdm


def box_iou(box, boxes):
    if len(boxes) == 0:
        return np.array([])
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-9
    return inter / union


def weighted_boxes_fusion(boxes, scores, iou_thr=0.55):
    if len(boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,))
    boxes = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float)

    order = scores.argsort()[::-1]
    boxes = boxes[order]
    scores = scores[order]

    used = np.zeros(len(boxes), dtype=bool)
    fused_boxes = []
    fused_scores = []

    for i in range(len(boxes)):
        if used[i]:
            continue
        ious = box_iou(boxes[i], boxes)
        group = np.where((ious >= iou_thr) & (~used))[0]

        group_boxes = boxes[group]
        group_scores = scores[group]

        weights = group_scores / (group_scores.sum() + 1e-9)
        fused_box = (group_boxes * weights[:, None]).sum(axis=0)
        fused_score = group_scores.mean() * min(1.0, len(group_scores) / 2.0)

        fused_boxes.append(fused_box)
        fused_scores.append(fused_score)
        used[group] = True

    return np.array(fused_boxes), np.array(fused_scores)


def load_gt(label_path, img_w, img_h):
    boxes = []
    if not label_path.exists():
        return np.zeros((0, 4))

    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        cls, xc, yc, w, h = map(float, line.split()[:5])
        if int(cls) != 0:
            continue
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])

    return np.asarray(boxes, dtype=float)


def draw_box(img, box, color, text=None, thickness=2):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if text:
        y_text = max(15, y1 - 5)
        cv2.putText(
            img,
            text,
            (x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--display-conf", type=float, default=0.15)
    parser.add_argument("--wbf-iou", type=float, default=0.55)
    parser.add_argument("--num-images", type=int, default=30)
    parser.add_argument("--positive-only", action="store_true")
    parser.add_argument("--out-dir", default="visualizations_boxes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.data, "r") as f:
        data = yaml.safe_load(f)

    root = Path(data.get("path", "."))
    val_images = root / data["val"]

    image_paths = sorted(list(val_images.glob("*.png")) + list(val_images.glob("*.jpg")) + list(val_images.glob("*.jpeg")))

    if args.positive_only:
        positives = []
        for p in image_paths:
            label_path = Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")
            if label_path.exists() and label_path.read_text().strip():
                positives.append(p)
        image_paths = positives

    sample = random.sample(image_paths, min(args.num_images, len(image_paths)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [YOLO(m) for m in args.models]

    for img_path in tqdm(sample, desc="Visualizing"):
        img = cv2.imread(str(img_path))
        if img is None:
            print("Could not read:", img_path)
            continue

        h, w = img.shape[:2]

        label_path = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
        gt_boxes = load_gt(label_path, w, h)

        all_boxes = []
        all_scores = []

        for model in models:
            results = model.predict(
                source=str(img_path),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=0.7,
                device=0,
                batch=1,
                half=True,
                verbose=False,
                stream=False,
            )
            r = results[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue

            boxes = r.boxes.xyxy.detach().cpu().numpy()
            scores = r.boxes.conf.detach().cpu().numpy()
            cls = r.boxes.cls.detach().cpu().numpy().astype(int)

            keep = cls == 0
            all_boxes.extend(boxes[keep].tolist())
            all_scores.extend(scores[keep].tolist())

        pred_boxes, pred_scores = weighted_boxes_fusion(all_boxes, all_scores, iou_thr=args.wbf_iou)

        vis = img.copy()

        for box in gt_boxes:
            draw_box(vis, box, (0, 255, 0), "GT", thickness=2)
        # Draw only the best matching prediction for each GT box
        used_pred = set()

        for gt_idx, gt_box in enumerate(gt_boxes):
            if len(pred_boxes) == 0:
                continue

            ious = box_iou(gt_box, pred_boxes)
            if len(ious) == 0:
                continue

            # best prediction for this GT
            pred_idx = int(np.argmax(ious))
            best_iou = float(ious[pred_idx])

            # avoid drawing the exact same prediction box multiple times if desired
            # comment this out if you want the same prediction to match multiple GTs
            if pred_idx in used_pred:
                continue
            used_pred.add(pred_idx)

            score = float(pred_scores[pred_idx])
            box = pred_boxes[pred_idx]

            draw_box(
                vis,
                box,
                (0, 0, 255),
                f"P {score:.3f} IoU {best_iou:.2f}",
                thickness=5
            )

        legend = "Green=GT | Red=Prediction"
        cv2.putText(vis, legend, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, legend, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)

        out_path = out_dir / f"{img_path.stem}_boxes.jpg"
        cv2.imwrite(str(out_path), vis)

    print(f"Saved visualizations to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
