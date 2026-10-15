"""The keypoint correctness check on made-up detections (numpy only)."""
import numpy as np

from fusion_scale.pose.keypoint_metrics import box_iou, compare, top_detection


def _out(boxes, scores, offset=0.0):
    n = len(scores)
    kp = np.zeros((n, 17, 3), dtype=np.float32)
    kp[..., 0] = 100 + offset
    kp[..., 1] = 200
    return {'boxes': np.asarray(boxes, dtype=np.float32).reshape(n, 4), 'scores': np.asarray(scores, dtype=np.float32),
            'keypoints': kp}


def test_top_detection_takes_the_most_confident_and_marks_empty_frames():
    boxes, kp = top_detection([_out([[0, 0, 10, 10], [5, 5, 50, 50]], [0.3, 0.9]), _out(np.zeros((0, 4)), [])])
    assert np.array_equal(boxes[0], [5, 5, 50, 50]) and np.isnan(boxes[1]).all() and np.isnan(kp[1]).all()


def test_box_iou_of_identical_and_disjoint_boxes():
    a = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    assert np.allclose(box_iou(a, b), [1.0, 0.0])


def test_compare_measures_keypoint_shift_and_mismatches():
    ref = top_detection([_out([[0, 0, 100, 100]], [0.9]), _out([[0, 0, 100, 100]], [0.9]), _out(np.zeros((0, 4)), [])])
    test = top_detection([_out([[0, 0, 100, 100]], [0.9], offset=3.0), _out([[200, 200, 300, 300]], [0.9]),
                          _out([[0, 0, 100, 100]], [0.9])])
    c = compare(ref, test)
    assert c['frames'] == 3 and c['found_by_one_only'] == 1 and c['different_person'] == 1
    assert abs(c['keypoint_px_mean'] - 3.0) < 1e-5
