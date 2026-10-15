"""Faster inference for torchvision's Keypoint R-CNN: compiled and TensorRT backbones, and capped detections.

Variants, timed on frames already on the GPU:
  eager       the model as torchvision provides it
  capped      eager, with the proposals and detections capped, for videos of one person
  compile     the backbone and feature pyramid compiled with torch.compile, for one batch size and frame size
  compile-cg  the same, with CUDA graphs ("reduce-overhead"), which cut the cost of launching kernels
  trt         the backbone and feature pyramid as a TensorRT engine (Torch-TensorRT), for one batch and size. Its
              precision is trt_mode: 'mixed' (fp32 module and input, TensorRT free to use FP16 kernels), 'fp32', or
              'half' (the module converted to fp16 first; on VIDIMU this build found no person in any frame)
In every variant the region heads stay in eager PyTorch, since their shapes depend on how many boxes each frame keeps.
The compiled backbones see one fixed shape: every frame is resized alike, and a short batch is padded to the compiled
batch size.
"""
import time
from collections import OrderedDict
from contextlib import nullcontext

import torch

VARIANTS = ('eager', 'capped', 'compile', 'compile-cg', 'trt')
CAPPED = dict(rpn_pre_nms_top_n_test=200, rpn_post_nms_top_n_test=50, box_detections_per_img=1)


def load_model(device, min_size=800, capped=False):
    """Keypoint R-CNN (ResNet-50 FPN, COCO weights). Size 800 is torchvision's default; 480 keeps 640x480 frames at
    their own size."""
    from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn
    max_size = 1333 if min_size == 800 else int(round(min_size * 4 / 3))
    model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT, min_size=min_size,
                                      max_size=max_size, **(CAPPED if capped else {}))
    return model.eval().to(device)


def load_frames(paths, per_video, device):
    """`per_video` frames from the start of each video, decoded once with TorchCodec on the CPU, as a list of float
    (3, height, width) tensors in [0, 1] on the GPU."""
    from torchcodec.decoders import VideoDecoder
    frames = []
    for path in paths:
        d = VideoDecoder(path)
        frames += list(d.get_frames_in_range(0, min(per_video, len(d))).data.to(device).float().div_(255).unbind(0))
    return frames


def autocast(precision):
    return torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()


class _AsTuple(torch.nn.Module):
    """The backbone with its feature maps as a tuple, which compilers handle more reliably than a dict."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return tuple(self.backbone(x).values())


class FixedBackbone(torch.nn.Module):
    """A backbone compiled for one input shape, made to look like the original to the model's heads: a short batch
    is padded to the compiled size, the input is cast to the compiled dtype, and float32 feature maps come back
    under the original names. Outputs are matched to names by their shapes, which are all different, so a
    compiler that reorders its outputs won't swap feature levels unnoticed."""

    def __init__(self, compiled, names, shape, dtype, out_channels, level_shapes=None):
        super().__init__()
        self.compiled, self.names, self.shape, self.dtype = compiled, list(names), tuple(shape), dtype
        self.out_channels = out_channels                        # the region heads read this attribute
        self.level_shapes = level_shapes                        # name: (channels, height, width)

    def forward(self, x):
        n = x.shape[0]
        if tuple(x.shape[1:]) != self.shape[1:]:
            raise RuntimeError(f'backbone input {tuple(x.shape)} does not match the compiled shape {self.shape}')
        if n < self.shape[0]:
            x = torch.cat([x, x.new_zeros((self.shape[0] - n, *x.shape[1:]))])
        outs = self.compiled(x.to(self.dtype))
        if self.level_shapes:
            by_shape = {tuple(o.shape[1:]): o for o in outs}
            outs = [by_shape[self.level_shapes[k]] for k in self.names]
        return OrderedDict((k, v[:n].float().clone()) for k, v in zip(self.names, outs))


def backbone_shape(model, frame, batch):
    """(batch, 3, height, width) of what the backbone receives: after the model's own resize and its padding."""
    with torch.inference_mode():
        images, _ = model.transform([frame])
    return (batch, *images.tensors.shape[1:])


def compile_backbone(model, variant, shape, precision, trt_mode='mixed'):
    """Replace model.backbone by a compiled one for `shape`; return the seconds spent, including the calls that
    trigger compilation or build the engine."""
    device = next(model.parameters()).device
    t = time.perf_counter()
    with torch.inference_mode():
        sample = model.backbone(torch.zeros(1, *shape[1:], device=device))
    names = list(sample.keys())
    level_shapes = {k: tuple(v.shape[1:]) for k, v in sample.items()}
    out_channels = model.backbone.out_channels
    if variant in ('compile', 'compile-cg'):
        mode = 'reduce-overhead' if variant == 'compile-cg' else 'default'
        compiled = torch.compile(_AsTuple(model.backbone), mode=mode, dynamic=False)   # graph breaks allowed
        dtype = torch.float32
    elif variant == 'trt':
        import torch_tensorrt
        module = _AsTuple(model.backbone).eval()
        if trt_mode == 'half':
            module, dtype, allowed = module.half(), torch.half, {torch.half}
        elif trt_mode == 'mixed':
            dtype, allowed = torch.float32, {torch.float32, torch.half}
        elif trt_mode == 'fp32':
            dtype, allowed = torch.float32, {torch.float32}
        else:
            raise ValueError(trt_mode)
        compiled = torch_tensorrt.compile(module, ir='dynamo', inputs=[torch_tensorrt.Input(shape=shape, dtype=dtype)],
                                          enabled_precisions=allowed, min_block_size=1)
    else:
        raise ValueError(variant)
    model.backbone = FixedBackbone(compiled, names, shape, dtype, out_channels, level_shapes)
    with torch.inference_mode(), autocast(precision):            # trigger compilation, as timed runs would see it
        for _ in range(3):
            model.backbone(torch.zeros(shape, device=device))
    torch.cuda.synchronize()
    return time.perf_counter() - t


def infer(model, frames, batch, precision):
    """The model's outputs for every frame, in order."""
    outs = []
    with torch.inference_mode(), autocast(precision):
        for i in range(0, len(frames), batch):
            outs += model(frames[i:i + batch])
    torch.cuda.synchronize()
    return outs


def time_inference(model, frames, batch, precision, rounds=3):
    """Seconds for each batch over `rounds` passes through `frames`, after one untimed pass."""
    times = []
    with torch.inference_mode(), autocast(precision):
        for r in range(rounds + 1):
            for i in range(0, len(frames), batch):
                torch.cuda.synchronize()
                t = time.perf_counter()
                model(frames[i:i + batch])
                torch.cuda.synchronize()
                if r:
                    times.append(time.perf_counter() - t)
    return times
