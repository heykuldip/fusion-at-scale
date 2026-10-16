![tests](https://github.com/heykuldip/fusion-at-scale/actions/workflows/tests.yml/badge.svg)
# fusion-at-scale

Measured checks for training and running activity-recognition models on synchronized video and wearable-sensor data
across GPUs: whether distributed training is correct and how it scales, how the data are synchronized and loaded, how
fast video becomes body-joint positions, and whether fusing joints with inertial (IMU) data helps. The data are the
public VIDIMU dataset; the runs used NVIDIA A100 and H100 GPUs.

| Experiment | Question | Code |
|---|---|---|
| DDP equivalence | Does one distributed data-parallel step match the same step on one device? | `fusion_scale/train/ddp_equivalence.py` |
| Scaling | How do throughput and time per step change from 1 to 4 GPUs, and where does the time go? | `fusion_scale/train/scaling.py`, `fusion_scale/bench/nccl_smoke.py` |
| Sampler checks | Does each epoch see every sample once across processes, and what do `drop_last` and `set_epoch` change? | `fusion_scale/data/sampler_checks.py` |
| Synchronized windows | Can video, body joints and IMU joint angles be cut into matching two-second windows, and how fast do they load? | `fusion_scale/data/sync_windows.py`, `scripts/build_windows.py`, `scripts/loader_timing.py` |
| Pose throughput | How fast does video become body-joint positions, and which stage limits it? | `fusion_scale/pose/pipeline.py`, `scripts/pose_throughput.py` |
| Fusion | Does intermediate fusion of joints and IMU data beat late fusion on subjects the models haven't seen? | `fusion_scale/fusion/`, `scripts/fusion_*.py` |
| Inference latency | How much faster can pose inference run, with its output unchanged? | `fusion_scale/pose/inference_variants.py`, `scripts/inference_latency.py` |

## Findings

- **DDP equivalence.** After one step on 2, 4 or 8 GPUs, every parameter matched the single-GPU step to within 2^-27
  (about 7.5e-9, at the level of fp32 rounding), on A100 40 GB, A100 80 GB and H100 GPUs. A deliberately wrong
  version that sums the processes' losses instead of averaging them failed the check.
- **Scaling.** At 4 GPUs, a 3-million-parameter model reached 86% (A100) and 89% (H100) of ideal weak scaling, and a
  101-million-parameter model 96% and 97%; fixed per-step costs limit the small model. The H100 was 2.0 times as fast
  as the A100 on the larger model and 1.6 times on the small one.
- **Sampler checks.** All 48 cases, with 2 to 4 processes and with and without `drop_last` and `set_epoch`, matched
  PyTorch's documented behaviour. Without `sampler.set_epoch(epoch)`, every epoch has the same order; combined with
  `drop_last`, the same samples are left out every epoch.
- **Synchronized windows.** 208 recordings of 16 subjects gave 5,637 two-second windows, split by subject. The
  dataset's own synchronization aligned video and IMU data within one frame in 92% of recordings. Data problems had to
  be fixed first, among them frames lost when seeking into a video and a trim recorded in video frames instead of
  sensor samples. Video decoding ran at about 420 frames a second per loader process, and 2,845 with 8.
- **Pose throughput.** torchvision's Keypoint R-CNN, as distributed, ran at about 80 frames a second on an A100 and
  130 on an H100. Decoding ran 12 to 30 times faster than the model, so the model is the limit. Independent processes
  on 2 and 4 GPUs of one node each reached 99% of the ideal total.
- **Fusion.** Leave-one-subject-out over 16 subjects, 3 seeds, macro-F1:

  | | Lower body (4 activities) | Upper body (9 activities) |
  |---|---|---|
  | Joints alone | **0.877** | 0.879 |
  | IMU alone | 0.732 | 0.919 |
  | Late fusion | 0.859 | **0.935** |
  | Intermediate fusion | 0.867 | 0.915 |
  | Intermediate minus late | +0.008 (95% CI -0.011 to +0.033; p = 0.98) | -0.020 (-0.033 to -0.008; p = 0.005) |

  For the upper body, late fusion scored highest, 0.020 above intermediate fusion; for the lower body, joints alone
  did as well as either fusion. Shuffled labels and the channel mask stayed at chance. A position-only control scored
  0.55 on the lower body: standing position carries information, which the joint features remove by centring on the
  hips.
- **Inference latency.** Compiling the backbone (+58%), keeping one detection per frame (+17 to 30%), a mixed-precision
  TensorRT engine (a further 10 to 15%) and the frames' own size took one A100 from about 80 to 318 frames a second,
  with keypoints within about one pixel of the original model's at the same size; an H100 reached 540. A build
  converted entirely to fp16 was the fastest but found no person in any frame, because values overflow fp16's range;
  comparing with eager inference caught it.

## Layout

```text
src/fusion_scale/    the package: train/, bench/, data/, models/, pose/, fusion/, and preflight.py
scripts/             entry points, reports, the dataset download and preflight.sh
slurm/               one job template per experiment
tests/               unit tests and smoke tests
requirements/        pinned versions for the two environments
environment/         creating and activating the pose environment
```

## Setup

Training, data and fusion experiments:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements/train.txt && pip install --no-deps -e .
bash scripts/download_vidimu.sh          # VIDIMU into data/vidimu (VIDIMU_DIR); "all" adds the full-size videos
```

Pose and inference experiments use a second environment with FFmpeg, TorchCodec and TensorRT:

```bash
bash environment/create_pose_env.sh      # needs conda
source environment/activate_pose.sh
```

## Tests

```bash
python -m pytest -m "not gpu"            # unit tests, and smoke tests of the distributed code on CPU processes
python -m pytest -m gpu                  # on a GPU node, in the pose environment, with VIDIMU_DIR set
```

The smoke tests run each experiment's real entry point end to end on tiny settings. Continuous integration runs the
linter and the CPU tests on every push.

## Running experiments

Each experiment has a job template in `slurm/` that requests resources only. Check it, then submit it:

```bash
SLURM_ARGS="-p <partition> -A <account>" bash scripts/preflight.sh scaling --submit
```

`preflight.sh` stops at the first failure:

1. the CPU tests;
2. `sbatch --test-only`, which checks the script and request (partition, GPUs, limits) without submitting;
3. in a short allocation with the job's GPUs, CPUs and memory, the environment, then two timed short runs of the real
   pipeline at the job's batch and model sizes;
4. the full run's time and memory, estimated from the short runs, against the job's `--time` and `--mem`;
5. with `--submit`, the job.

A job's settings come from the environment, so the estimate is for the run that will be submitted, for example
`NS="1 2" SCALING_ARGS="--width 1024 --depth 8" bash scripts/preflight.sh scaling`. Results go to
`results/<experiment>/`; the `scripts/*_report.py` scripts summarize them.

## Data

VIDIMU is not included. `scripts/download_vidimu.sh` downloads it from Zenodo (record 15075076) and checks every
file. It is licensed CC BY 4.0: Martinez-Zarzuela et al. (2023), Scientific Data 10, 648,
https://doi.org/10.1038/s41597-023-02554-9.

## License and acknowledgments

Released under the GNU General Public License v3.0; see `LICENSE`.
Experiments ran on an HPC cluster, with computing resources provided by the Office of Research Computing at
George Mason University (https://orc.gmu.edu).
