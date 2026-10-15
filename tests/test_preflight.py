"""The preflight checks' arithmetic, parsing and Slurm calls, without Slurm."""
import json
import os
import stat
import sys

import pytest

from fusion_scale import preflight as pf


@pytest.mark.parametrize('value,seconds', [('30', 1800), ('5:30', 330), ('1:00:00', 3600), ('2-00', 172800),
                                           ('0-02:00:00', 7200), ('1-12:30', 131400), ('0-00:20:00', 1200)])
def test_parse_time(value, seconds):
    assert pf.parse_time(value) == seconds


def test_format_time_rounds_up_to_the_minute():
    assert pf.format_time(61) == '0-00:02:00' and pf.format_time(90000) == '1-01:00:00'


@pytest.mark.parametrize('value,gb', [('128G', 128), ('64g', 64), ('1T', 1024), ('2048', 2), ('2048M', 2)])
def test_parse_mem(value, gb):
    assert pf.parse_mem(value) == gb * 1024 ** 3


def test_sbatch_options_and_gpus(tmp_path):
    job = tmp_path / 'job.sbatch'
    job.write_text('#!/bin/bash\n#SBATCH --gres=gpu:4\n#SBATCH -c 32\n#SBATCH --mem=128G\n#SBATCH --time 0-02:00:00\n'
                   'set -e\n#SBATCH --mem=1G\n')
    opts = pf.sbatch_options(str(job))
    assert opts == {'gres': 'gpu:4', 'cpus-per-task': '32', 'mem': '128G', 'time': '0-02:00:00'}
    assert pf.requested_gpus(opts) == 4
    assert pf.requested_gpus({'gres': 'gpu:a100:2'}) == 2 and pf.requested_gpus({}) == 0


def test_fit_recovers_startup_and_rate():
    startup, per_unit, noisy = pf.fit([(10, 25.0), (20, 45.0), (40, 85.0)])
    assert startup == pytest.approx(5.0) and per_unit == pytest.approx(2.0) and not noisy
    assert pf.fit([(10, 30.0)]) == (0.0, 3.0, False)
    assert pf.fit([(10, 30.0), (20, 25.0)]) == (25.0, 0.5, True)   # noisy: the faster run, and the spread per unit


def test_estimate_and_compare():
    needed = pf.estimate([(10, 25.0), (20, 45.0)], full_units=1000, launches=4, peak_rss_bytes=8 * 1024 ** 3)
    assert needed['time_s'] == pytest.approx((4 * 5 + 1000 * 2) * 1.5)
    assert pf.compare(needed, {'time': '1:00:00', 'mem': '16G'}) == []
    levels = [level for level, _ in pf.compare(needed, {'time': '0:30:00', 'mem': '8G'})]
    assert levels == ['fail', 'fail']
    assert [level for level, _ in pf.compare(needed, {'time': '1-00:00:00', 'mem': '256G'})] == ['warn', 'warn']


def test_noisy_runs_warn_instead_of_failing_a_startup_bound_job():
    needed = pf.estimate([(8, 96.6), (16, 90.0)], full_units=320, launches=1)     # mostly start-up
    assert needed['noisy_fit'] and needed['time_s'] < 3600
    levels = [level for level, _ in pf.compare(needed, {'time': '0-01:00:00', 'mem': '8G'})]
    assert 'fail' not in levels and 'warn' in levels


def test_run_records_time_memory_and_exit_code(tmp_path):
    out = tmp_path / 'r.json'
    assert pf.main(['run', '--out', str(out), '--', sys.executable, '-c', 'x = bytearray(50_000_000)']) == 0
    record = json.loads(out.read_text())
    assert record['exit_code'] == 0 and record['seconds'] > 0 and record['peak_rss_bytes'] > 40_000_000
    assert pf.main(['run', '--out', str(out), '--', sys.executable, '-c', 'raise SystemExit(3)']) == 3


@pytest.mark.skipif(not os.path.isdir('/proc'), reason='needs /proc')
def test_run_counts_processes_running_at_once(tmp_path):
    out = tmp_path / 'r.json'
    child = 'import time; x = bytearray(60_000_000); time.sleep(1.5)'
    parent = (f'import subprocess, sys; ps = [subprocess.Popen([sys.executable, "-c", {child!r}]) for _ in range(3)]; '
              '[p.wait() for p in ps]')
    assert pf.main(['run', '--out', str(out), '--', sys.executable, '-c', parent]) == 0
    assert json.loads(out.read_text())['peak_rss_bytes'] > 3 * 60_000_000     # all three together, not one


def test_estimate_command_fails_when_the_request_is_too_small(tmp_path):
    job = tmp_path / 'job.sbatch'
    job.write_text('#!/bin/bash\n#SBATCH --time=0-00:10:00\n#SBATCH --mem=4G\n')
    for name, seconds in (('a', 25.0), ('b', 45.0)):
        record = {'exit_code': 0, 'seconds': seconds, 'peak_rss_bytes': 2 ** 30}
        (tmp_path / f'{name}.json').write_text(json.dumps(record))
    measured = [f"{tmp_path / 'a.json'}:10", f"{tmp_path / 'b.json'}:20"]
    assert pf.main(['estimate', '--job', str(job), '--measured', *measured, '--full-units', '1000']) == 1
    assert pf.main(['estimate', '--job', str(job), '--measured', *measured, '--full-units', '100']) == 0


def test_test_only_reads_slurms_answer(tmp_path, monkeypatch):
    fake = tmp_path / 'sbatch'
    fake.write_text('#!/bin/sh\necho "sbatch: Job 123 to start at 2030-01-01T00:00:00 using 32 processors" >&2\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv('PATH', f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    ok, message = pf.test_only('job.sbatch')
    assert ok and 'to start at' in message
    fake.write_text('#!/bin/sh\necho "sbatch: error: Invalid generic resource (gres) specification" >&2\nexit 1\n')
    assert pf.test_only('job.sbatch')[0] is False


def test_env_check_needs_enough_gpus():
    torch = pytest.importorskip('torch')
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    assert pf.check_env(0)['torch'] == torch.__version__
    with pytest.raises(RuntimeError):
        pf.check_env(n + 1)


def test_job_settings_and_shell_output(tmp_path, capsys):
    job = tmp_path / 'job.sbatch'
    job.write_text('#!/bin/bash\n#SBATCH --gres=gpu:2\n#SBATCH --time=0-01:00:00\n#SBATCH --mem=8G\nset -e\n'
                   ': "${NS:=1 2 4}" "${REPEATS:=3}"\n: "${ARGS:=}"\necho ${NOT_A_SETTING:=x}\n')
    assert pf.job_settings(str(job)) == {'NS': '1 2 4', 'REPEATS': '3', 'ARGS': ''}
    assert pf.main(['options', '--job', str(job), '--shell']) == 0
    out = capsys.readouterr().out
    assert 'JOB_GPUS=2' in out and ': "${NS:=1 2 4}"; export NS' in out


def test_job_settings_read_multi_line_defaults():
    import glob
    root = os.path.join(os.path.dirname(__file__), '..', 'slurm')
    settings = pf.job_settings(os.path.join(root, 'pose_throughput.sbatch'))
    configs = [line for line in settings['CONFIGS'].split('\n') if line.strip()]
    assert len(configs) == 15 and all(c.startswith('--decoder') for c in configs)
    for job in glob.glob(os.path.join(root, '*.sbatch')):                 # every template declares time and memory
        opts = pf.sbatch_options(job)
        assert pf.parse_time(opts['time']) > 0 and pf.parse_mem(opts['mem']) > 0
        assert opts['output'].startswith('logs/') and 'partition' not in opts and 'account' not in opts
