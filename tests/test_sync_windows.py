"""Synchronized windows on a tiny synthetic VIDIMU tree (tests/vidimu_fixture.py): real file formats, known answers."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from vidimu_fixture import (  # noqa: E402
    LOCKED,
    LOWER_COLS,
    MEASURED,
    MOT_COLS,
    UPPER_COLS,
    frame_index,
    make_fixture,
)

from fusion_scale.data.vidimu import inventory, read_cuts, read_joints, read_mot  # noqa: E402

from fusion_scale.data.sync_windows import (  # noqa: E402
    Window,
    alignment_lag,
    build_windows,
    read_index,
    split_by_subject,
    synchronized_streams,
    write_index,
)


@pytest.fixture(scope='module')
def tree(tmp_path_factory):
    root = str(tmp_path_factory.mktemp('vidimu'))
    return root, make_fixture(root)


def test_inventory_finds_every_trial_with_no_problems(tree):
    root, answers = tree
    trials, problems = inventory(root)
    if not any(t.video for t in trials):                       # without PyAV the fixture has no videos
        problems = [p for p in problems if 'no video' not in p]
    assert sorted(t.key for t in trials) == sorted(answers) and problems == []
    by_key = {t.key: t for t in trials}
    assert by_key['S90_A05_T01'].video_cut == 3 and by_key['S90_A05_T01'].joints_sync
    assert by_key['S91_A02_T01'].imu_cut == 1 and by_key['S91_A02_T01'].mot_sync
    assert by_key['S90_A01_T01'].joints_sync is None and by_key['S90_A01_T01'].mot_sync is None


def test_readers_keep_every_row_and_clean_the_names(tree):
    root, _ = tree
    trial = {t.key: t for t in inventory(root)[0]}['S91_A02_T01']
    joints, names = read_joints(trial.joints)
    assert joints.shape == (120, 34, 3) and names[-1] == 'right_thumb_tip'       # the leading space is gone
    times, cols, values = read_mot(trial.mot_sync)
    assert cols == MOT_COLS and times[0] == 0.0 and round(times[1], 10) == 0.04   # N-pose row kept, times unchanged
    assert read_cuts(os.path.join(root, 'analysis', 'analysis', 'videoandimusync', 'infoToSync.csv'))[
        ('S90', 'A05', 'T01', 'csv')] == 3


def test_read_mot_rejects_a_header_that_does_not_match_its_rows(tmp_path):
    bad = tmp_path / 'bad.mot'
    bad.write_text('endheader\ntime\ta_b\n0\t1\t2\n')             # 2 names, 3 values a row
    with pytest.raises(ValueError, match='2 names'):
        read_mot(str(bad))


def test_split_by_subject_is_fixed_and_disjoint():
    subjects = [f'S{i}' for i in range(40, 56)]
    a, b = split_by_subject(subjects, seed=0), split_by_subject(subjects, seed=0)
    assert a == b and sorted(a.values()).count('test') == 2 and sorted(a.values()).count('val') == 2


def test_index_round_trip(tmp_path):
    w = Window('S90', 'A01', 'T01', 0, 'lower', 0.0, 2.0, 0, 60, 0, 0, 100, 1.0)
    write_index([w], {'S90': 'train'}, str(tmp_path / 'idx.csv'))
    windows, splits = read_index(str(tmp_path / 'idx.csv'))
    assert windows == [w] and splits == {'S90': 'train'}


def test_fixture_videos_decode_to_the_right_frames(tree):
    pytest.importorskip('av')
    from fusion_scale.data.window_dataset import decode_frames
    root, _ = tree
    trial = {t.key: t for t in inventory(root)[0]}['S90_A05_T01']
    frames = decode_frames(trial.video, 3 + 50, 10)               # synchronized row 50 is .mp4 frame 53
    assert [frame_index(f) for f in frames] == list(range(53, 63))


def test_streams_start_together_without_the_npose_row(tree):
    """For every trial, imu_time starts at 0 and steps by 0.02; the IMU's first row is the first motion row (for
    S91_A02_T01's synchronized file, the row stamped 0.04 s); joints has 120 rows, imu 195.
    """
    root, answers = tree
    trials = inventory(root)[0]

    for trial in trials:
        streams = synchronized_streams(trial)

        assert streams['joints'].shape == (120, 34, 3)
        assert streams['imu'].shape == (195, len(MOT_COLS))

        assert streams['imu_time'][0] == pytest.approx(0.0)
        np.testing.assert_allclose(
            np.diff(streams['imu_time']),
            1.0 / 50.0,
        )

        # The source .mot still contains its N-pose row.  Row 1 is therefore
        # the first motion row, even when its retained timestamp is 0.04 s.
        mot_path = trial.mot_sync or trial.mot
        times, cols, raw = read_mot(mot_path)

        assert times[1] == pytest.approx(
            answers[trial.key]['first_motion_file_time']
        )

        measured = answers[trial.key]['measured']
        measured_idx = [cols.index(name) for name in measured]

        # synchronized_streams() drops row 0 (N-pose), so its row 0 should be
        # the .mot file's first motion row.
        np.testing.assert_allclose(
            streams['imu'][0, measured_idx],
            raw[1, measured_idx],
        )


def test_video_offset_follows_the_cut(tree):
    """S90_A05_T01's video_offset is 3, the others' 0; and a window's video_frame_start is its frame_start plus that
    offset.
    """
    root, answers = tree
    trials = inventory(root)[0]

    for trial in trials:
        streams = synchronized_streams(trial)

        assert streams['video_offset'] == answers[trial.key]['video_offset']

        if trial.key == 'S90_A05_T01':
            assert streams['video_offset'] == 3
        else:
            assert streams['video_offset'] == 0

        windows, _ = build_windows(trial, streams)

        for win in windows:
            assert (
                win.video_frame_start
                == win.frame_start + streams['video_offset']
            )


def test_unmeasured_channels_are_nan_and_lost_frames_flagged(tree):
    """A lower-body trial's upper-body columns are NaN (and the reverse), as are constant columns; frames 40-43 are
    flagged as not tracked, the others as tracked.
    """

    root, _ = tree
    trials = inventory(root)[0]

    for trial in trials:
        streams = synchronized_streams(trial)

        names = streams['imu_names']
        imu = streams['imu']

        expected_measured = set(MEASURED[trial.body])

        for j, name in enumerate(names):
            if name in expected_measured:
                # The fixture makes every genuinely measured coordinate vary.
                assert np.isfinite(imu[:, j]).all(), (
                    trial.key, name
                )
            else:
                # Includes:
                #   - the entire uninstrumented body half, even though its
                #     inverse-kinematics coordinates drift;
                #   - constant/locked coordinates such as pelvis translations.
                assert np.isnan(imu[:, j]).all(), (
                    trial.key, name
                )

        # Check that body-half membership matters, not only
        # whether the raw column happens to vary.
        mot_path = trial.mot_sync or trial.mot
        _, raw_names, raw = read_mot(mot_path)

        # Remove N-pose: synchronized_streams() does the same.
        raw = raw[1:]

        other_half = UPPER_COLS if trial.body == 'lower' else LOWER_COLS

        drifting_other_half = [
            name
            for name in other_half
            if name not in LOCKED
            and np.ptp(raw[:, raw_names.index(name)]) > 0
        ]

        # The fixture deliberately gives these nonzero variation.
        assert drifting_other_half

        for name in drifting_other_half:
            j = names.index(name)
            assert np.isnan(imu[:, j]).all(), (
                trial.key,
                name,
                'varies but is on the uninstrumented body half',
            )

        expected_tracked = np.ones(len(streams['joints']), dtype=bool)
        expected_tracked[40:44] = False

        np.testing.assert_array_equal(
            streams['tracked'],
            expected_tracked,
        )


def test_windows_never_mix_trials_and_fit_both_streams(tree):
    """Every window holds 60 frames and 100 IMU rows, inside its own trial's streams; windows that would run past either
    stream's end are dropped and counted (with 4 s of video and 3.9 s of IMU data, two fit and one is dropped).
    """
    root, _ = tree
    trials = inventory(root)[0]

    all_windows = []

    for trial in trials:
        streams = synchronized_streams(trial)
        windows, dropped = build_windows(
            trial,
            streams,
            length_s=2.0,
            stride_s=1.0,
        )

        # Starts 0 and 1 s fit both streams.  Start 2 s fits the 4.0-s
        # video exactly but exceeds the 3.9-s IMU stream.
        assert len(windows) == 2
        assert dropped == 1

        assert [w.start_s for w in windows] == [0.0, 1.0]
        assert [w.frame_start for w in windows] == [0, 30]
        assert [w.imu_start for w in windows] == [0, 50]

        for win in windows:
            assert win.subject == trial.subject
            assert win.activity == trial.activity
            assert win.trial == trial.trial

            assert win.frame_stop - win.frame_start == 60
            assert win.imu_stop - win.imu_start == 100

            assert 0 <= win.frame_start < win.frame_stop <= len(
                streams['joints']
            )
            assert 0 <= win.imu_start < win.imu_stop <= len(
                streams['imu']
            )

        all_windows.extend(windows)

    # Three trials, two valid windows each.
    assert len(all_windows) == 6


def test_alignment_lag_is_zero_after_sync_and_the_check_can_fail(tree):
    """alignment_lag() returns 0 for each synchronized trial. Negative control: given S90_A05_T01's original, uncut
    joints, it returns the cut (3 frames), so the check can detect misalignment.
    """
    root, _ = tree
    trials = inventory(root)[0]
    by_key = {t.key: t for t in trials}

    # Positive control: authors' synchronized streams are aligned.
    for trial in trials:
        streams = synchronized_streams(trial)
        assert alignment_lag(trial, streams) == 0

    # Negative control: S90_A05_T01 originally begins three video frames
    # too early.  Put those original, uncut joints back into the stream.
    trial = by_key['S90_A05_T01']
    streams = synchronized_streams(trial)

    original_joints, original_names = read_joints(trial.joints)

    uncut = dict(streams)
    uncut['joints'] = original_joints
    uncut['joint_names'] = original_names
    uncut['video_offset'] = 0

    # In the fixture BodyTrack's default lost-subject pose is all zeros.
    uncut['tracked'] = np.any(
        original_joints != 0.0,
        axis=(1, 2),
    )

    assert alignment_lag(trial, uncut) == 3
