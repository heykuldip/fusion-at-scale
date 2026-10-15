"""A tiny synthetic VIDIMU tree for the window tests: real file formats, known answers.

Three trials, 4 seconds of video (120 frames at 30 a second) and 3.9 s of IMU data (50 Hz), so the streams end apart
as VIDIMU's do:
  S90_A01_T01  lower body, no cut
  S90_A05_T01  upper body, the video stream cut by 3 frames (a synchronized .csv; the .mp4 itself isn't cut)
  S91_A02_T01  lower body, the IMU stream cut by 1 video frame, which removes floor(1 x 50/30) = 1 .mot row (a
               synchronized .mot that keeps its original timestamps)
The joint of interest follows one known angle theta(t), so the alignment check has a right answer. BodyTrack "loses"
the subject in frames 40-43 of every trial: those rows hold one fixed default pose. As in VIDIMU's real files, the
.mot has columns no IMU measures in that activity: some locked at a constant (pelvis_ty at 0.94), and the other body
half's angles drifting slightly (inverse kinematics moves coordinates nothing constrains). Only MEASURED[body] are
measurements. With PyAV installed, each trial also gets a small .mp4 whose frames encode their own index.
"""
import os

import numpy as np

JOINTS = ['pelvis', 'left_hip', 'right_hip', 'torso', 'left_knee', 'right_knee', 'neck', 'left_ankle', 'right_ankle',
          'left_big_toe', 'right_big_toe', 'left_small_toe', 'right_small_toe', 'left_heel', 'right_heel', 'nose',
          'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow',
          'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky_knuckle', 'right_pinky_knuckle', 'left_middle_tip',
          'right_middle_tip', 'left_index_knuckle', 'right_index_knuckle', 'left_thumb_tip', 'right_thumb_tip']
LOWER_COLS = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz', 'hip_flexion_r',
              'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
              'knee_angle_l']
UPPER_COLS = ['lumbar_extension', 'lumbar_bending', 'lumbar_rotation', 'arm_flex_r', 'arm_add_r', 'arm_rot_r',
              'elbow_flex_r', 'pro_sup_r', 'arm_flex_l', 'arm_add_l', 'arm_rot_l', 'elbow_flex_l', 'pro_sup_l']
MOT_COLS = LOWER_COLS + UPPER_COLS
LOCKED = {'pelvis_tx': 0.0, 'pelvis_ty': 0.94, 'pelvis_tz': 0.0}          # constant in every motion row
MEASURED = {'lower': [c for c in LOWER_COLS if c not in LOCKED], 'upper': UPPER_COLS}
FRAMES, IMU_ROWS, LOST = 120, 195, range(40, 44)
TRIALS = {  # key: (activity's chain (from, joint, to), .mot column, video cut, IMU cut)
    'S90_A01_T01': (('left_hip', 'left_knee', 'left_ankle'), 'knee_angle_l', 0, 0),
    'S90_A05_T01': (('right_shoulder', 'right_elbow', 'right_wrist'), 'elbow_flex_r', 3, 0),
    'S91_A02_T01': (('right_hip', 'right_knee', 'right_ankle'), 'knee_angle_r', 0, 1),
}


def theta(t):
    """The joint of interest's angle, degrees, at time t seconds on the synchronized timeline."""
    return 40 + 25 * np.sin(2 * np.pi * 0.4 * t) + 10 * np.sin(2 * np.pi * 1.1 * t + 0.7)


def _joints(chain, times):
    """(frames, 34, 3): every joint fixed, except the chain's last joint, which swings to make the angle theta."""
    base = np.array([[100.0 * (i % 6), 120.0 * (i // 6), 6000.0] for i in range(len(JOINTS))])
    out = np.repeat(base[None], len(times), axis=0)
    a, b, c = (JOINTS.index(n) for n in chain)
    for f, t in enumerate(times):
        out[f, b] = out[f, a] + (0, 450, 0)                           # first segment points straight down (+y)
        phi = np.radians(theta(t) + 5.0)                               # the video's angle reads 5 degrees high
        out[f, c] = out[f, b] + 420 * np.array([np.sin(phi), np.cos(phi), 0.0])
    return out


def _write_csv(path, joints):
    header = []
    for n in JOINTS:
        sep = ' ' if n == 'right_thumb_tip' else ''                   # VIDIMU's last three names start with a space
        header += [f'{sep}{n}_x', f'{sep}{n}_y', f'{sep}{n}_z']
    with open(path, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in joints.reshape(len(joints), -1):
            f.write(','.join(f'{v:.1f}' for v in row) + '\n')


def _write_mot(path, name, times, rows):
    with open(path, 'w') as f:
        f.write(f'inDegrees=yes\nname={name}\nDataType=double\nversion=3\nOpenSimVersion=4.3-fixture\nendheader\n')
        f.write('\t'.join(['time'] + MOT_COLS) + '\n')
        for t, r in zip(times, rows):
            f.write('\t'.join([repr(round(float(t), 10))] + [repr(float(v)) for v in r]) + '\n')


def frame_index(frame_rgb):
    """The index a fixture video's frame encodes, in two grey halves (grey has no colour to lose in compression):
    the left half is 4 x (index mod 60) + 2, the right half 100 x (index div 60) + 50."""
    left, right = frame_rgb[:, 4:28].mean(), frame_rgb[:, 36:60].mean()
    return int(round((left - 2) / 4)) + 60 * int(round((right - 50) / 100))


def _write_video(path, frames):
    import av
    with av.open(path, 'w') as c:
        s = c.add_stream('libx264', rate=30)
        s.width, s.height, s.pix_fmt, s.options = 64, 48, 'yuv420p', {'g': '12', 'crf': '0'}
        for i in range(frames):
            img = np.zeros((48, 64, 3), dtype=np.uint8)
            img[:, :32], img[:, 32:] = 4 * (i % 60) + 2, 100 * (i // 60) + 50
            for p in s.encode(av.VideoFrame.from_ndarray(img, format='rgb24')):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def make_fixture(root: str) -> dict:
    """Write the tree under root; return the known answers by trial key."""
    ds = os.path.join(root, 'dataset', 'dataset')
    answers, cut_rows = {}, ['Subject,Activity,Trial,File,Type,CutFrames,OrigRmse,TheoRmse']
    try:
        import av  # noqa: F401
        with_video = True
    except ImportError:
        with_video = False
    for key, (chain, column, video_cut, imu_cut) in TRIALS.items():
        s, a, t = key.split('_')
        upper = a not in ('A01', 'A02', 'A03', 'A04')
        for d in ('videoandimus', 'videoandimusync'):
            os.makedirs(os.path.join(ds, d, s), exist_ok=True)
        # video: the recording starts video_cut frames early; the synchronized .csv drops them
        video_times = (np.arange(FRAMES + video_cut) - video_cut) / 30.0
        joints = _joints(chain, video_times)
        joints[[f + video_cut for f in LOST]] = 0.0                    # BodyTrack's default pose: every joint at 0
        _write_csv(os.path.join(ds, 'videoandimus', s, f'{key}.csv'), joints)
        _write_csv(os.path.join(ds, 'videoandimus', s, f'{key}_Npose.csv'), joints[:5])
        if video_cut:
            _write_csv(os.path.join(ds, 'videoandimusync', s, f'{key}.csv'), joints[video_cut:])
            cut_rows += [f'{s},{a},{t},D:/x/{key}.mp4,mp4,{video_cut},9.9,1.1',
                         f'{s},{a},{t},D:/x/{key}.csv,csv,{video_cut},9.9,1.1']
        # IMU: an N-pose row at time 0, then motion rows from 0.02 s; the recording starts imu_cut rows early
        n = IMU_ROWS + imu_cut
        times = np.arange(n + 1) * 0.02
        motion_t = (np.arange(n) - imu_cut) * 0.02                     # synchronized time of each motion row
        rows = np.zeros((n + 1, len(MOT_COLS)))
        rows[0, MOT_COLS.index('pelvis_ty')], rows[0, MOT_COLS.index('arm_add_r')] = 0.94, 90.0
        measured = MEASURED['upper' if upper else 'lower']
        for j, col in enumerate(measured):
            rows[1:, MOT_COLS.index(col)] = 3.0 * np.sin(2 * np.pi * 0.3 * motion_t + j)
        for j, col in enumerate(LOWER_COLS if upper else UPPER_COLS):     # the other half drifts: not data
            if col not in LOCKED:
                rows[1:, MOT_COLS.index(col)] = 0.4 * np.sin(2 * np.pi * 0.05 * motion_t + j)
        for col, value in LOCKED.items():
            rows[1:, MOT_COLS.index(col)] = value
        rows[1:, MOT_COLS.index(column)] = theta(motion_t)
        _write_mot(os.path.join(ds, 'videoandimus', s, f'ik_{key}.mot'), f'ik_{key}', times, rows)
        if imu_cut:                                                    # keeps the N-pose row and the original times
            keep = [0] + list(range(1 + imu_cut, n + 1))
            _write_mot(os.path.join(ds, 'videoandimusync', s, f'ik_{key}.mot'), f'ik_{key}', times[keep], rows[keep])
            cut_rows += [f'{s},{a},{t},D:/x/{key}.raw,raw,{imu_cut},9.9,1.1',
                         f'{s},{a},{t},D:/x/{key}.mot,mot,{imu_cut},9.9,1.1']
        if with_video:
            vdir = os.path.join(root, 'videosmallsize', 'videosmallsize', 'videosoriginal', s)
            os.makedirs(vdir, exist_ok=True)
            _write_video(os.path.join(vdir, f'{key}.mp4'), FRAMES + video_cut)
        answers[key] = {'video_offset': video_cut, 'imu_cut': imu_cut, 'frames': FRAMES, 'imu_rows': IMU_ROWS,
                        'lost_frames': list(LOST), 'measured': measured, 'column': column,
                        'body': 'upper' if upper else 'lower',
                        'first_motion_file_time': round(0.02 * (1 + imu_cut), 10)}
    adir = os.path.join(root, 'analysis', 'analysis', 'videoandimusync')
    os.makedirs(adir, exist_ok=True)
    with open(os.path.join(adir, 'infoToSync.csv'), 'w') as f:
        f.write('\n'.join(cut_rows) + '\n')
    return answers


if __name__ == "__main__":
    import sys
    print(make_fixture(sys.argv[1]).keys())
