"""Smoke tests: each experiment's real entry point, end to end, on tiny settings.

The distributed ones run on CPU processes with the gloo backend, so they also run in continuous integration. The
pose and inference ones need a CUDA GPU, torchvision, TorchCodec and the VIDIMU videos (VIDIMU_DIR), and are skipped
without them. Run only these with: python -m pytest -m smoke
"""
import glob
import json
import os
import subprocess
import sys

import pytest

from vidimu_fixture import make_fixture

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(ROOT, 'src'), os.environ.get('PYTHONPATH', '')]),
           OMP_NUM_THREADS='1')
pytestmark = pytest.mark.smoke


def run(args, cwd):
    p = subprocess.run(args, cwd=cwd, env=ENV, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-4000:]
    return p


def torchrun(n, module, *args, cwd):
    return run([sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={n}', '-m', module,
                *args], cwd)


def test_nccl_smoke_on_cpu(tmp_path):
    out = tmp_path / 'nccl.json'
    torchrun(2, 'fusion_scale.bench.nccl_smoke', '--backend', 'gloo', '--size-mb', '1', '--iters', '2', '--warmup', '1',
             '--out', str(out), cwd=tmp_path)
    assert json.loads(out.read_text())['correct'] is True


def test_ddp_equivalence_on_cpu(tmp_path):
    out = tmp_path / 'ddp.json'
    torchrun(2, 'fusion_scale.train.ddp_equivalence', '--backend', 'gloo', '--out', str(out), cwd=tmp_path)
    assert json.loads(out.read_text())['pass'] is True


def test_scaling_on_cpu(tmp_path):
    out = tmp_path / 'scaling.json'
    torchrun(2, 'fusion_scale.train.scaling', '--backend', 'gloo', '--steps', '4', '--warmup', '1', '--width', '32',
             '--depth', '1', '--per-gpu-batch', '16', '--n-samples', '256', '--workers', '0', '--out', str(out),
             cwd=tmp_path)
    r = json.loads(out.read_text())
    assert r['n_gpus'] == 2 and r['steps'] == 4 and r['samples_per_s'] > 0


def test_sampler_checks_on_cpu(tmp_path):
    out = tmp_path / 'sampler.json'
    torchrun(3, 'fusion_scale.data.sampler_checks', '--backend', 'gloo', '--out', str(out), cwd=tmp_path)
    cases = json.loads(out.read_text())['cases']
    assert cases and all(c['matches_docs'] for c in cases)


def test_build_windows_and_loader_on_the_fixture(tmp_path):
    data = tmp_path / 'vidimu'
    make_fixture(str(data))
    out = tmp_path / 'windows'
    run([sys.executable, os.path.join(ROOT, 'scripts', 'build_windows.py'), '--root', str(data), '--out', str(out)],
        tmp_path)
    summary = json.loads(next(out.glob('summary-*.json')).read_text())
    assert summary['windows'] == 6 and summary['windows_dropped'] == 3 and summary['failures'] == []
    pytest.importorskip('av')
    index = next(out.glob('windows-*.csv'))
    run([sys.executable, os.path.join(ROOT, 'scripts', 'loader_timing.py'), '--index', str(index), '--root', str(data),
         '--workers', '0', '--batch', '2', '--video-batch', '2', '--batches', '1', '--out', str(out)], tmp_path)
    timing = json.loads(next(out.glob('loader-*.json')).read_text())
    assert {r['with_video'] for r in timing['runs']} == {False, True}


def _gpu_with_videos():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('no CUDA GPU')
    pytest.importorskip('torchvision')
    pytest.importorskip('torchcodec')
    from fusion_scale.pose.pipeline import FULL_VIDEOS
    root = os.environ.get('VIDIMU_DIR')
    if not root or not glob.glob(os.path.join(root, FULL_VIDEOS, 'S40', '*.mp4')):
        pytest.skip('set VIDIMU_DIR to the VIDIMU dataset, with the full-size videos')
    return root


@pytest.mark.gpu
def test_pose_throughput_on_a_gpu(tmp_path):
    root = _gpu_with_videos()
    run([sys.executable, os.path.join(ROOT, 'scripts', 'pose_throughput.py'), '--root', root, '--subjects', 'S40',
         '--decoder', 'cpu', '--batch', '4', '--warmup', '4', '--frames', '16', '--keypoints', str(tmp_path / 'kp'),
         '--out', str(tmp_path)], tmp_path)
    r = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert r['result']['frames'] == 16 and r['rates']['end_to_end_fps'] > 0


@pytest.mark.gpu
def test_inference_latency_on_a_gpu(tmp_path):
    root = _gpu_with_videos()
    run([sys.executable, os.path.join(ROOT, 'scripts', 'inference_latency.py'), '--root', root, '--variant', 'capped',
         '--size', '480', '--batch', '4', '--frames', '16', '--rounds', '1', '--out', str(tmp_path)], tmp_path)
    r = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert 'error' not in r and r['frames_per_s'] > 0
