import argparse
import yaml
from pathlib import Path
import numpy as np
import torch
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

        # Conservative score: average score boosted slightly by model agreement.
        fused_score = group_scores.mean() * min(1.0, len(group_scores) / 2.0)

        fused_boxes.append(fused_box)
        fused_scores.append(fused_score)
        used[group] = True

    return np.array(fused_boxes), np.array(fused_scores)


def load_yolo_gt(label_path, img_w, img_h):
    boxes = []
    if not label_path.exists():
        return np.zeros((0, 4))

    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls = int(float(parts[0]))
        if cls != 0:
            continue
        xc, yc, w, h = map(float, parts[1:5])
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])

    return np.asarray(boxes, dtype=float)


def compute_ap(preds, gts, iou_thr=0.5):
    all_scores = []
    all_tp = []
    all_fp = []
    total_gt = 0

    for img_id in preds:
        pred_boxes, pred_scores = preds[img_id]
        gt_boxes = gts.get(img_id, np.zeros((0, 4)))
        total_gt += len(gt_boxes)

        if len(pred_boxes) == 0:
            continue

        order = pred_scores.argsort()[::-1]
        pred_boxes = pred_boxes[order]
        pred_scores = pred_scores[order]

        matched = np.zeros(len(gt_boxes), dtype=bool)

        for box, score in zip(pred_boxes, pred_scores):
            all_scores.append(score)

            if len(gt_boxes) == 0:
                all_tp.append(0)
                all_fp.append(1)
                continue

            ious = box_iou(box, gt_boxes)
            best_idx = int(np.argmax(ious))
            best_iou = ious[best_idx]

            if best_iou >= iou_thr and not matched[best_idx]:
                all_tp.append(1)
                all_fp.append(0)
                matched[best_idx] = True
            else:
                all_tp.append(0)
                all_fp.append(1)

    if total_gt == 0 or len(all_scores) == 0:
        return 0.0

    order = np.argsort(all_scores)[::-1]
    tp = np.asarray(all_tp)[order]
    fp = np.asarray(all_fp)[order]

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / (total_gt + 1e-9)
    precision = tp_cum / (tp_cum + fp_cum + 1e-9)

    # COCO-style interpolation over 101 recall points
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        p = precision[recall >= r].max() if np.any(recall >= r) else 0
        ap += p / 101

    return float(ap)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--wbf-iou", type=float, default=0.55)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    with open(args.data, "r") as f:
        data = yaml.safe_load(f)

    root = Path(data.get("path", "."))
    val_images = root / data["val"]
    if not val_images.exists():
        raise FileNotFoundError(f"Val image directory not found: {val_images}")

    image_paths = sorted(list(val_images.glob("*.png")) + list(val_images.glob("*.jpg")) + list(val_images.glob("*.jpeg")))
    if args.max_images:
        image_paths = image_paths[:args.max_images]

    print(f"Images: {len(image_paths)}")
    print("Models:")
    for m in args.models:
        print(" -", m)

    gts = {}
    all_model_preds = []

    for model_path in args.models:
        model = YOLO(model_path)
        model_preds = {}

        for img_path in tqdm(image_paths, total=len(image_paths), desc=f"Predict {Path(model_path).parent.parent.name}"):
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
            h, w = r.orig_shape
            img_id = img_path.stem

            if img_id not in gts:
                label_path = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
                gts[img_id] = load_yolo_gt(label_path, w, h)

            if r.boxes is None or len(r.boxes) == 0:
                model_preds[img_id] = (np.zeros((0, 4)), np.zeros((0,)))
                continue

            boxes = r.boxes.xyxy.detach().cpu().numpy()
            scores = r.boxes.conf.detach().cpu().numpy()
            cls = r.boxes.cls.detach().cpu().numpy().astype(int)

            keep = cls == 0
            model_preds[img_id] = (boxes[keep], scores[keep])

        all_model_preds.append(model_preds)

    ensemble_preds = {}

    for img_path in tqdm(image_paths, desc="Fuse"):
        img_id = img_path.stem
        boxes_all = []
        scores_all = []

        for mp in all_model_preds:
            boxes, scores = mp.get(img_id, (np.zeros((0, 4)), np.zeros((0,))))
            boxes_all.extend(boxes.tolist())
            scores_all.extend(scores.tolist())

        fused_boxes, fused_scores = weighted_boxes_fusion(
            boxes_all,
            scores_all,
            iou_thr=args.wbf_iou,
        )

        ensemble_preds[img_id] = (fused_boxes, fused_scores)

    ap25 = compute_ap(ensemble_preds, gts, iou_thr=0.25)
    ap50 = compute_ap(ensemble_preds, gts, iou_thr=0.5)
    aps = [compute_ap(ensemble_preds, gts, iou_thr=t) for t in np.arange(0.5, 1.0, 0.05)]
    ap5095 = float(np.mean(aps))

    print("\nENSEMBLE RESULTS")
    print(f"AP25:     {ap25:.4f}")
    print(f"AP50:     {ap50:.4f}")
    print(f"AP50-95:  {ap5095:.4f}")
    print("\nAP by IoU threshold:")
    for t, ap in zip(np.arange(0.5, 1.0, 0.05), aps):
        print(f"IoU {t:.2f}: {ap:.4f}")


if __name__ == "__main__":
    main()
