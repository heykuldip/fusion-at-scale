"""Correctness check for inference variants, in numpy: how far keypoints move from a reference's on the same frames."""
import numpy as np


def _np(x):
    return x.detach().float().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x, dtype=np.float32)


def top_detection(outputs):
    """For each frame, the most confident person's box (4,) and keypoints' x, y (17, 2); NaN for frames with no
    detection. Coordinates are in the original frame's pixels, whatever size the model resized it to."""
    boxes = np.full((len(outputs), 4), np.nan, dtype=np.float32)
    keypoints = np.full((len(outputs), 17, 2), np.nan, dtype=np.float32)
    for i, o in enumerate(outputs):
        scores = _np(o['scores'])
        if scores.size:
            k = int(np.argmax(scores))
            boxes[i] = _np(o['boxes'])[k]
            keypoints[i] = _np(o['keypoints'])[k][:, :2]
    return boxes, keypoints


def box_iou(a, b):
    """Intersection over union of matching rows of two (n, 4) arrays of x1, y1, x2, y2 boxes."""
    w = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    h = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = w * h
    union = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    with np.errstate(invalid='ignore', divide='ignore'):
        return inter / union


def compare(ref, test):
    """How `test` (boxes, keypoints) differs from `ref` on the same frames. Keypoint error is the mean distance over
    the 17 keypoints, in pixels, on frames where both found the same person (boxes overlapping at least half)."""
    (rb, rk), (tb, tk) = ref, test
    found_r, found_t = np.isfinite(rb[:, 0]), np.isfinite(tb[:, 0])
    both = found_r & found_t
    iou = np.full(len(rb), np.nan)
    iou[both] = box_iou(rb[both], tb[both])
    same = both & (iou >= 0.5)
    err = np.linalg.norm(rk[same] - tk[same], axis=2).mean(axis=1) if same.any() else np.array([])
    return {'frames': int(len(rb)), 'found_by_one_only': int((found_r ^ found_t).sum()),
            'different_person': int((both & (iou < 0.5)).sum()),
            'keypoint_px_mean': float(err.mean()) if err.size else None,
            'keypoint_px_p95': float(np.percentile(err, 95)) if err.size else None,
            'keypoint_px_max': float(err.max()) if err.size else None}
