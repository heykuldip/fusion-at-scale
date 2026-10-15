"""VIDIMU's files: where each trial's streams are, and a reader for each format. numpy only.

Layout of VIDIMU 2.0.0, unpacked under ROOT:
  dataset/dataset/videoandimus/S##/     S##_A##_T##.csv       BodyTrack's 34 joints from the video, one row per frame
                                        S##_A##_T##_Npose.csv the same during the N-pose calibration
                                        ik_S##_A##_T##.mot    OpenSim's joint angles from the IMUs, 50 Hz
                                        S##_A##_T##.raw/.sto  the IMUs' quaternions (not used)
  dataset/dataset/videoandimusync/S##/  the files the authors changed to synchronize the streams, only where needed
  analysis/analysis/videoandimusync/infoToSync.csv   each change: Subject, Activity, Trial, File, Type, CutFrames, ...
  videos{mall,full}size/.../videosoriginal/S##/S##_A##_T##.mp4   the trial's video, without the skeleton drawn on

The constants follow the dataset's paper: Martinez-Zarzuela et al., Scientific Data 10, 648 (2023).
"""
import csv
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

VIDEO_FPS = 30.0
IMU_HZ = 50.0
ACTIVITIES = tuple(f'A{i:02d}' for i in range(1, 14))       # 13 classes; label = index in this tuple
LOWER_BODY = frozenset({'A01', 'A02', 'A03', 'A04'})          # the rest use the upper body
VIDEO_DIRS = {'small': 'videosmallsize/videosmallsize/videosoriginal',
              'full': 'videosfullsize/videosfullsize/videosoriginal'}

# The paper's Table 4: each activity's joint of interest, used by the authors to synchronize the streams. Given as
# two BodyTrack segments (from joint, to joint), whose angle matches the named .mot column. Used by the alignment check.
_KNEE_L = (('left_hip', 'left_knee'), ('left_knee', 'left_ankle'), 'knee_angle_l')
_KNEE_R = (('right_hip', 'right_knee'), ('right_knee', 'right_ankle'), 'knee_angle_r')
_ELBOW_R = (('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'), 'elbow_flex_r')
_ELBOW_L = (('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'), 'elbow_flex_l')
_SHOULDER_R = (('neck', 'torso'), ('right_shoulder', 'right_elbow'), 'arm_flex_r')
_SHOULDER_L = (('neck', 'torso'), ('left_shoulder', 'left_elbow'), 'arm_flex_l')
JOINT_OF_INTEREST = {'A01': _KNEE_L, 'A03': _KNEE_L, 'A02': _KNEE_R, 'A04': _KNEE_R,
                     'A05': _ELBOW_R, 'A09': _ELBOW_R, 'A06': _ELBOW_L,
                     'A07': _SHOULDER_R, 'A10': _SHOULDER_R, 'A11': _SHOULDER_R, 'A13': _SHOULDER_R,
                     'A08': _SHOULDER_L, 'A12': _SHOULDER_L}


@dataclass(frozen=True)
class Trial:
    """One subject's one activity: every file that could hold each stream, and the authors' cuts."""
    subject: str
    activity: str
    trial: str
    joints: str                   # videoandimus/S##/S##_A##_T##.csv
    joints_sync: Optional[str]    # videoandimusync's version, if the authors cut the video stream
    mot: Optional[str]            # videoandimus/S##/ik_S##_A##_T##.mot
    mot_sync: Optional[str]       # videoandimusync's version, if the authors cut the IMU stream
    video: Optional[str]          # the trial's .mp4 in videosoriginal, if found
    video_cut: int                # frames cut from the video stream (infoToSync.csv, csv and mp4); 0 if none
    imu_cut: int                  # the IMU stream's cut (infoToSync.csv, mot), in VIDEO frames: the authors removed
                                  # floor(imu_cut x 50 / 30) .mot rows; 0 if none

    @property
    def key(self) -> str:
        return f'{self.subject}_{self.activity}_{self.trial}'

    @property
    def body(self) -> str:
        return 'lower' if self.activity in LOWER_BODY else 'upper'

    @property
    def label(self) -> int:
        return ACTIVITIES.index(self.activity)


def read_joints(path: str):
    """BodyTrack's joints: an array (frames, joints, 3) in mm, and the joint names.

    The header's names can carry spaces (VIDIMU's last three do), so they are stripped; the columns are expected
    in x, y, z triplets."""
    with open(path, newline='') as f:
        header = [h.strip() for h in next(csv.reader(f))]
    if len(header) % 3 or any(not h.endswith(f'_{axis}') for h, axis in zip(header, 'xyz' * (len(header) // 3))):
        raise ValueError(f'{path}: columns are not x, y, z triplets')
    names = [h[:-2] for h in header[0::3]]
    data = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2) if os.path.getsize(path) else np.empty((0, 0))
    if data.size == 0:
        return np.empty((0, len(names), 3)), names
    if data.shape[1] != len(header):
        raise ValueError(f'{path}: {len(header)} names in the header, {data.shape[1]} values a row')
    return data.reshape(len(data), len(names), 3), names


def read_mot(path: str):
    """OpenSim's .mot or .sto: times (rows,), the column names after 'time', and values (rows, columns).

    Every data row is returned, the N-pose row included. A header whose names don't match the rows' widths is an
    error, not something to guess around."""
    with open(path, newline='') as f:
        lines = [line.rstrip('\r\n') for line in f]
    try:
        end = next(i for i, line in enumerate(lines) if line.strip().lower() == 'endheader')
    except StopIteration:
        raise ValueError(f'{path}: no endheader line') from None
    names = [c.strip() for c in lines[end + 1].split('\t')]
    rows = [line.split('\t') for line in lines[end + 2:] if line.strip()]
    if names[0].lower() != 'time':
        raise ValueError(f'{path}: first column is {names[0]!r}, not time')
    widths = {len(r) for r in rows}
    if rows and widths != {len(names)}:
        raise ValueError(f'{path}: {len(names)} names in the header, rows have {sorted(widths)} values')
    values = np.array(rows, dtype=float) if rows else np.empty((0, len(names)))
    return values[:, 0], names[1:], values[:, 1:]


def read_cuts(path: str) -> dict:
    """infoToSync.csv as {(subject, activity, trial, type): frames cut}; type is mp4, csv, raw or mot."""
    cuts = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            cuts[(r['Subject'], r['Activity'], r['Trial'], r['Type'])] = int(float(r['CutFrames']))
    return cuts


def inventory(root: str, video_size: str = 'small'):
    """Every trial with both video and IMUs, and a list of problems found on the way (missing files, a changed file
    without a cut, mismatched cuts). Problems are reported, not fixed silently."""
    ds = os.path.join(root, 'dataset', 'dataset')
    both, sync = os.path.join(ds, 'videoandimus'), os.path.join(ds, 'videoandimusync')
    cuts_path = os.path.join(root, 'analysis', 'analysis', 'videoandimusync', 'infoToSync.csv')
    cuts = read_cuts(cuts_path) if os.path.exists(cuts_path) else {}
    videos = os.path.join(root, VIDEO_DIRS[video_size])
    pattern = re.compile(r'^(S\d+)_(A\d+)_(T\d+)\.csv$')        # skips the _Npose files
    trials, problems = [], [] if cuts else [f'no {cuts_path}']

    def present(path):
        return path if os.path.exists(path) else None

    for subject in sorted(os.listdir(both)):
        folder = os.path.join(both, subject)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            m = pattern.match(name)
            if not m:
                continue
            s, a, t = m.groups()
            key = f'{s}_{a}_{t}'
            csv_cut, mp4_cut = cuts.get((s, a, t, 'csv'), 0), cuts.get((s, a, t, 'mp4'), 0)
            trial = Trial(subject=s, activity=a, trial=t, joints=os.path.join(folder, name),
                          joints_sync=present(os.path.join(sync, s, name)),
                          mot=present(os.path.join(folder, f'ik_{key}.mot')),
                          mot_sync=present(os.path.join(sync, s, f'ik_{key}.mot')),
                          video=present(os.path.join(videos, s, f'{key}.mp4')),
                          video_cut=csv_cut, imu_cut=cuts.get((s, a, t, 'mot'), 0))
            if a not in ACTIVITIES:
                problems.append(f'{key}: unknown activity')
            if trial.mot is None:
                problems.append(f'{key}: no .mot')
            if trial.video is None:
                problems.append(f'{key}: no video in {VIDEO_DIRS[video_size]}')
            if csv_cut != mp4_cut:
                problems.append(f'{key}: cuts differ for the .csv ({csv_cut}) and the .mp4 ({mp4_cut})')
            if (trial.joints_sync is not None) != (csv_cut > 0):
                problems.append(f'{key}: synchronized .csv and cut disagree '
                                f'({trial.joints_sync is not None}, {csv_cut})')
            if (trial.mot_sync is not None) != (trial.imu_cut > 0):
                problems.append(f'{key}: synchronized .mot and cut disagree '
                                f'({trial.mot_sync is not None}, {trial.imu_cut})')
            trials.append(trial)
    return trials, problems
