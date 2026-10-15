"""Checks before submitting a full job: the environment, a timed short run of the real pipeline, and whether the job's
request covers the full run.

    python -m fusion_scale.preflight env --gpus 4
    python -m fusion_scale.preflight options --job slurm/scaling.sbatch --shell
    python -m fusion_scale.preflight run --out short1.json -- python scripts/pose_throughput.py --frames 64
    python -m fusion_scale.preflight estimate --job slurm/scaling.sbatch --measured short1.json:30 short2.json:60 \\
        --full-units 2640 --launches 12
    python -m fusion_scale.preflight test-only --job slurm/scaling.sbatch -- -p <partition>

The estimate fits wall time = start-up + units x time per unit to one or more short runs, scales it to the full run
(start-up once per launch), applies a safety margin, and compares it, and the short runs' peak memory, with the job
script's --time and --mem. A request that is too small fails; one more than four times what's needed is flagged.
"""
import argparse
import json
import math
import os
import re
import resource
import shlex
import subprocess
import sys
import time

OVERSIZED = 4.0


def parse_time(value):
    """Seconds in a Slurm time limit: M, M:S, H:M:S, D-H, D-H:M or D-H:M:S."""
    value = value.strip()
    days = 0
    if '-' in value:
        d, value = value.split('-', 1)
        days = int(d)
        parts = [int(p) for p in value.split(':')]
        parts += [0] * (3 - len(parts))
        hours, minutes, seconds = parts
    else:
        parts = [int(p) for p in value.split(':')]
        if len(parts) == 1:
            hours, minutes, seconds = 0, parts[0], 0
        elif len(parts) == 2:
            hours, (minutes, seconds) = 0, parts
        else:
            hours, minutes, seconds = parts
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def format_time(seconds):
    """A Slurm time limit, D-HH:MM:SS, rounded up to the minute."""
    minutes = max(1, math.ceil(seconds / 60))
    days, minutes = divmod(minutes, 24 * 60)
    hours, minutes = divmod(minutes, 60)
    return f'{days}-{hours:02d}:{minutes:02d}:00'


def parse_mem(value):
    """Bytes in a Slurm memory request: a number with an optional K, M, G or T suffix (M if none)."""
    m = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*([KMGT]?)B?\s*', value, re.I)
    if not m:
        raise ValueError(f'unrecognized memory request {value!r}')
    scale = {'K': 1, 'M': 2, 'G': 3, 'T': 4}.get(m.group(2).upper() or 'M')
    return int(float(m.group(1)) * 1024 ** scale)


def format_mem(n_bytes):
    """A Slurm memory request in whole gigabytes, rounded up."""
    return f'{max(1, math.ceil(n_bytes / 1024 ** 3))}G'


SHORT = {'-t': 'time', '-c': 'cpus-per-task', '-N': 'nodes', '-n': 'ntasks', '-p': 'partition', '-A': 'account',
         '-a': 'array', '-J': 'job-name', '-G': 'gpus'}


def sbatch_options(path):
    """The #SBATCH options at the top of a job script, as {long name: value}."""
    options = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#!'):
                continue
            if not line.startswith('#SBATCH'):
                if line.strip() and not line.startswith('#'):
                    break                                   # Slurm stops reading options at the first command
                continue
            words = shlex.split(line[len('#SBATCH'):], comments=True)
            if not words:
                continue
            key, _, value = words[0].partition('=')
            if not value and len(words) > 1:
                value = words[1]
            key = SHORT.get(key, key.lstrip('-'))
            options[key] = value
    return options


def job_settings(path):
    """The job script's settings with defaults, written as `: "${NAME:=default}"` (a default can span lines), as
    {NAME: default}."""
    text = open(path).read()
    settings = {}
    for m in re.finditer(r'\$\{([A-Z_][A-Z0-9_]*):=([^}]*)\}', text):
        line_start = text.rfind('\n', 0, m.start()) + 1
        if text[line_start:m.start()].lstrip().startswith(':'):
            settings[m.group(1)] = m.group(2)
    return settings


def requested_gpus(options):
    """GPUs per node requested by --gres=gpu[:type]:N or --gpus N; 0 if none."""
    gres = options.get('gres', '')
    m = re.search(r'gpu(?::[^:,]+)?:(\d+)', gres)
    if m:
        return int(m.group(1))
    if re.fullmatch(r'gpu(:[^:,]+)?', gres):
        return 1
    return int(options['gpus'].split(':')[-1]) if 'gpus' in options else 0


def _tree_rss(root):
    """Resident memory, in bytes, of a process and all its descendants, from /proc (Linux); None elsewhere."""
    if not os.path.isdir('/proc'):
        return None
    children = {}
    for entry in os.listdir('/proc'):
        if entry.isdigit():
            try:
                with open(f'/proc/{entry}/stat') as f:
                    stat = f.read()
            except OSError:
                continue
            ppid = int(stat[stat.rindex(')') + 2:].split()[1])
            children.setdefault(ppid, []).append(int(entry))
    total, todo = 0, [root]
    while todo:
        pid = todo.pop()
        todo += children.get(pid, [])
        try:
            with open(f'/proc/{pid}/status') as f:
                total += next((int(line.split()[1]) * 1024 for line in f if line.startswith('VmRSS:')), 0)
        except OSError:
            pass
    return total


def measure(command, interval=0.2):
    """Run a command; return its wall time in seconds, exit code, and peak resident memory. The peak is the largest
    total over the command and all its processes at once (sampled every `interval` seconds on Linux), so a job of
    several processes, such as a torchrun launch, counts them together, as Slurm's memory limit does."""
    t = time.perf_counter()
    proc = subprocess.Popen(command)
    peak = 0
    while True:
        peak = max(peak, _tree_rss(proc.pid) or 0)
        try:
            code = proc.wait(timeout=interval)
            break
        except subprocess.TimeoutExpired:
            continue
    seconds = time.perf_counter() - t
    largest_one = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
    return {'command': command, 'exit_code': code, 'seconds': round(seconds, 3),
            'peak_rss_bytes': max(peak, largest_one)}


def fit(points):
    """Least-squares start-up and time per unit from (units, seconds) pairs, and whether the runs were too noisy to
    show time growing with units. One point gives no start-up time. Noisy runs give the fastest run's time as start-up
    and the runs' spread as the time per unit."""
    if len(points) == 1:
        units, seconds = points[0]
        return 0.0, seconds / units, False
    n = len(points)
    mu = sum(u for u, _ in points) / n
    ms = sum(s for _, s in points) / n
    var = sum((u - mu) ** 2 for u, _ in points)
    if var == 0:
        raise ValueError('the short runs need different unit counts')
    per_unit = sum((u - mu) * (s - ms) for u, s in points) / var
    if per_unit <= 0:
        return min(s for _, s in points), -per_unit, True
    return max(ms - per_unit * mu, 0.0), per_unit, False


def estimate(points, full_units, launches=1, peak_rss_bytes=0, margin=1.5, mem_margin=1.25):
    """Time and memory the full run needs, with safety margins."""
    startup, per_unit, noisy = fit(points)
    seconds = (max(launches, 1) * startup + full_units * per_unit) * margin      # a run starts at least once
    return {'startup_s': round(startup, 2), 'per_unit_s': per_unit, 'noisy_fit': noisy,
            'full_units': full_units, 'launches': launches,
            'time_s': round(seconds, 1), 'time': format_time(seconds),
            'mem_bytes': int(peak_rss_bytes * mem_margin), 'mem': format_mem(peak_rss_bytes * mem_margin)}


def compare(needed, options):
    """Problems with the job's request against what's needed: ('fail', message) or ('warn', message)."""
    problems = []
    if needed.get('noisy_fit'):
        problems.append(('warn', "the short runs' times didn't grow with their size, so the estimate takes their "
                                 'spread as the time per unit; time is mostly start-up'))
    if 'time' not in options:
        problems.append(('fail', 'the job script sets no --time'))
    else:
        have = parse_time(options['time'])
        if have < needed['time_s']:
            problems.append(('fail', f"--time {options['time']} is shorter than the {needed['time']} needed"))
        elif have > OVERSIZED * needed['time_s']:
            problems.append(('warn', f"--time {options['time']} is over {OVERSIZED:g} times the {needed['time']} "
                                     'needed'))
    if needed['mem_bytes']:
        if 'mem' not in options:
            problems.append(('warn', 'the job script sets no --mem, so the partition default applies'))
        else:
            have = parse_mem(options['mem'])
            if have < needed['mem_bytes']:
                problems.append(('fail', f"--mem {options['mem']} is less than the {needed['mem']} needed"))
            elif have > OVERSIZED * needed['mem_bytes']:
                problems.append(('warn', f"--mem {options['mem']} is over {OVERSIZED:g} times the {needed['mem']} "
                                         'needed'))
    return problems


def check_env(gpus=0):
    """Versions and GPUs; raises RuntimeError with fewer usable GPUs than `gpus`."""
    import platform

    import torch
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    info = {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpus': [torch.cuda.get_device_name(i) for i in range(n)]}
    for i in range(n):                                       # a small matmul per GPU checks that kernels run
        a = torch.randn(256, 256, device=f'cuda:{i}')
        (a @ a).sum().item()
    if n < gpus:
        raise RuntimeError(f'{n} usable GPUs, {gpus} needed')
    return info


def test_only(job, extra=()):
    """sbatch --test-only: Slurm validates the script and request without submitting. Returns (ok, Slurm's message)."""
    p = subprocess.run(['sbatch', '--test-only', *extra, job], capture_output=True, text=True)
    message = (p.stderr + p.stdout).strip()
    return p.returncode == 0 and 'to start at' in message, message


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('env', help='versions and GPUs')
    e.add_argument('--gpus', type=int, default=0)
    o = sub.add_parser('options', help="a job script's #SBATCH options")
    o.add_argument('--job', required=True)
    o.add_argument('--shell', action='store_true',
                   help="as shell assignments: JOB_TIME, JOB_MEM, JOB_CPUS, JOB_GPUS, and the job's settings")
    r = sub.add_parser('run', help='run a command and record its wall time and peak memory')
    r.add_argument('--out', required=True)
    r.add_argument('cmd', nargs=argparse.REMAINDER)
    s = sub.add_parser('estimate', help='the full run against the job request')
    s.add_argument('--job', required=True)
    s.add_argument('--measured', nargs='+', required=True, help='short-run records as FILE:UNITS')
    s.add_argument('--full-units', type=float, required=True)
    s.add_argument('--launches', type=int, default=1)
    s.add_argument('--margin', type=float, default=1.5)
    t = sub.add_parser('test-only', help='sbatch --test-only')
    t.add_argument('--job', required=True)
    t.add_argument('extra', nargs=argparse.REMAINDER)
    a = p.parse_args(argv)

    if a.command == 'env':
        try:
            print(json.dumps(check_env(a.gpus), indent=1))
        except RuntimeError as err:
            print(f'environment: {err}', file=sys.stderr)
            return 1
        return 0
    if a.command == 'options':
        opts = sbatch_options(a.job)
        if a.shell:
            print(f"JOB_TIME={shlex.quote(opts.get('time', ''))} JOB_MEM={shlex.quote(opts.get('mem', ''))} "
                  f"JOB_CPUS={shlex.quote(opts.get('cpus-per-task', '1'))} JOB_GPUS={requested_gpus(opts)}")
            for name, default in job_settings(a.job).items():         # defaults, unless already set
                print(f': "${{{name}:={default}}}"; export {name}')
        else:
            print(json.dumps(opts, indent=1))
        return 0
    if a.command == 'run':
        cmd = a.cmd[1:] if a.cmd[:1] == ['--'] else a.cmd
        record = measure(cmd)
        with open(a.out, 'w') as f:
            json.dump(record, f, indent=1)
        print(json.dumps({k: record[k] for k in ('exit_code', 'seconds', 'peak_rss_bytes')}))
        return record['exit_code']
    if a.command == 'estimate':
        points, peak = [], 0
        for item in a.measured:
            path, _, units = item.rpartition(':')
            record = json.load(open(path))
            if record['exit_code'] != 0:
                print(f'{path}: the short run failed (exit {record["exit_code"]})', file=sys.stderr)
                return 1
            points.append((float(units), record['seconds']))
            peak = max(peak, record['peak_rss_bytes'])
        needed = estimate(points, a.full_units, a.launches, peak, a.margin)
        problems = compare(needed, sbatch_options(a.job))
        print(json.dumps({'needed': needed, 'problems': problems}, indent=1))
        return 1 if any(level == 'fail' for level, _ in problems) else 0
    if a.command == 'test-only':
        extra = a.extra[1:] if a.extra[:1] == ['--'] else a.extra
        ok, message = test_only(a.job, extra)
        print(message)
        return 0 if ok else 1
    return 2


if __name__ == '__main__':
    sys.exit(main())
