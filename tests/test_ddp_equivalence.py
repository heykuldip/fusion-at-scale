"""One DDP step on two CPU processes (gloo) matches one full-batch step on one process; a summed loss doesn't."""
import json

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fusion_scale.train.ddp_equivalence import (ddp_step, make_global_batch, make_model, max_abs_diff,
                                                reference_step, set_determinism)

WORLD, PER_PROCESS, LR, SEED = 2, 16, 0.1, 0


def _worker(rank, init_file, scale, out):
    dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank, world_size=WORLD)
    set_determinism(SEED)
    x, y = make_global_batch(SEED, PER_PROCESS * WORLD, torch.device('cpu'))
    shard = slice(rank * PER_PROCESS, (rank + 1) * PER_PROCESS)
    model = make_model(SEED)
    if scale != 1:                                           # a wrong loss: the shards' losses summed, not averaged
        model.register_full_backward_pre_hook(lambda m, g: tuple(t * scale for t in g))
    grads, params = ddp_step(model, x[shard], y[shard], LR, device_ids=None)
    if rank == 0:
        ref_grads, ref_params = reference_step(make_model(SEED), x, y, LR)
        with open(out, 'w') as f:
            json.dump({'dgrad': max_abs_diff(ref_grads, grads), 'dparam': max_abs_diff(ref_params, params)}, f)
    dist.destroy_process_group()


def _run(tmp_path, scale=1):
    out = str(tmp_path / f'diff-{scale}.json')
    mp.spawn(_worker, args=(str(tmp_path / f'init-{scale}'), scale, out), nprocs=WORLD, join=True)
    with open(out) as f:
        return json.load(f)


def test_ddp_step_matches_the_full_batch_step(tmp_path):
    d = _run(tmp_path)
    assert d['dgrad'] <= 1e-6 and d['dparam'] <= 1e-6, d


def test_a_summed_loss_fails_the_check(tmp_path):
    d = _run(tmp_path, scale=WORLD)
    assert d['dgrad'] > 1e-3, d
