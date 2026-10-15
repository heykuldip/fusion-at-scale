"""The pose pipeline's plain parts, on any machine: the video list, keeping the top person, and writing keypoints."""
import numpy as np
import pytest

from fusion_scale.pose.pipeline import KeypointWriter, top_person, video_list


def test_video_list_skips_npose_clips_and_filters(tmp_path):
    d = tmp_path / 'videosfullsize' / 'videosfullsize' / 'videosoriginal' / 'S40'
    d.mkdir(parents=True)
    for name in ('S40_A01_T01.mp4', 'S40_A01_T01_Npose.mp4', 'S40_A05_T02.mp4'):
        (d / name).write_bytes(b'')
    assert [p.split('/')[-1] for p in video_list(str(tmp_path))] == ['S40_A01_T01.mp4', 'S40_A05_T02.mp4']
    assert video_list(str(tmp_path), activities=['A05'])[0].endswith('S40_A05_T02.mp4')
    assert video_list(str(tmp_path), subjects=['S41']) == []


def test_top_person_keeps_the_most_confident_and_marks_empty_frames():
    keypoints = np.arange(2 * 17 * 3, dtype=np.float32).reshape(2, 17, 3)
    outputs = [{'scores': np.array([0.2, 0.9]), 'keypoints': keypoints, 'keypoints_scores': np.ones((2, 17))},
               {'scores': np.array([]), 'keypoints': np.zeros((0, 17, 3)), 'keypoints_scores': np.zeros((0, 17))}]
    out = top_person(outputs)
    assert out.shape == (2, 17, 3)
    assert np.array_equal(out[0, :, :2], keypoints[1, :, :2]) and np.all(out[0, :, 2] == 1)
    assert np.isnan(out[1]).all()


def test_writer_puts_chunks_back_in_order(tmp_path):
    w = KeypointWriter(str(tmp_path))
    w.add('S40_A01_T01', 2, np.full((2, 17, 3), 2.0))
    w.add('S40_A01_T01', 0, np.full((2, 17, 3), 0.0))
    assert w.close() == 1
    saved = np.load(tmp_path / 'S40_A01_T01.npy')
    assert saved.shape == (4, 17, 3) and saved[0, 0, 0] == 0 and saved[3, 0, 0] == 2


def test_unknown_decoder_is_refused():
    pytest.importorskip('torchcodec')
    from fusion_scale.pose.pipeline import make_decoder
    with pytest.raises(ValueError, match='unknown decoder'):
        make_decoder('x.mp4', 'gpu')
