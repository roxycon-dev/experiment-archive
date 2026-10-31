"""NativePyramid-SP v3: direct multiscale semantics plus zero-start residuals.

Only the locked ImageNet W48 initialization is accepted. auxiliary_logits is
the single semantic prediction S; logits is U(S) + D. No weights are downloaded.
"""
from __future__ import annotations

import hashlib
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

W48_SHA256 = "d2186c55782308392ae14cf179ac4840143b09abdbba81aa3f9274c795ebc960"
BACKBONE_TENSORS = 1830
CLASSIFICATION_TENSORS = 126
FUSION_RESIZES = 31
DECODER_PARAMETERS = 246132
MODEL_PARAMETERS = 65571252
ARCHITECTURE_VERSION = "NativePyramid-SP-v3"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def resize(x, size):
    # All network and prediction interpolation has exactly this geometry.
    dtype = x.dtype
    if dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False).to(dtype)


class BilinearUpsample(nn.Module):
    def __init__(self, scale):
        super().__init__()
        if float(scale) != int(scale) or scale < 1:
            raise ValueError("Expected an integer HRNet fusion scale")
        self.scale = int(scale)

    def forward(self, x):
        return resize(x, (x.shape[-2] * self.scale, x.shape[-1] * self.scale))


def replace_fusion_resize(module):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Upsample):
            setattr(module, name, BilinearUpsample(child.scale_factor))
            count += 1
        else:
            count += replace_fusion_resize(child)
    return count


class W48Features(nn.Module):
    def __init__(self, pretrained, checkpointing=True):
        super().__init__()
        import timm
        actual_sha256 = file_sha256(pretrained)
        if actual_sha256 != W48_SHA256:
            raise ValueError("Expected the locked ImageNet W48 file, not task weights")
        self.net = timm.create_model("hrnet_w48", pretrained=False, features_only=True,
                                     feature_location="")
        self.resize_count = replace_fusion_resize(self.net)
        if self.resize_count != FUSION_RESIZES:
            raise RuntimeError(f"Expected 31 HRNet fusion resizes, got {self.resize_count}")
        state = torch.load(pretrained, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        own = self.net.state_dict()
        if len(own) != BACKBONE_TENSORS:
            raise RuntimeError(f"Expected 1830 backbone tensors, got {len(own)}")
        missing = [name for name, value in own.items()
                   if name not in state or state[name].shape != value.shape
                   or state[name].dtype != value.dtype]
        if missing:
            raise ValueError(f"Pretrained backbone key/shape/dtype mismatch: {missing[:10]}")
        excluded = sorted(set(state) - set(own))
        classification_prefixes = ("incre_modules.", "downsamp_modules.", "final_layer.",
                                   "classifier.")
        if len(excluded) != CLASSIFICATION_TENSORS or any(
                not name.startswith(classification_prefixes) for name in excluded):
            raise ValueError("Expected exactly 126 explicitly excluded classification tensors")
        self.net.load_state_dict({name: state[name] for name in own}, strict=True)
        mismatched = [name for name, value in self.net.state_dict().items()
                      if value.dtype != state[name].dtype or not torch.equal(value, state[name])]
        if mismatched:
            raise RuntimeError(f"Pretrained per-key tensor identity failed: {mismatched[:10]}")
        self.initialization = {
            "architecture": ARCHITECTURE_VERSION,
            "file_sha256": actual_sha256,
            "verified_backbone_tensors": len(own),
            "verified_backbone_key_sha256": hashlib.sha256(
                "\n".join(sorted(own)).encode("utf-8")).hexdigest(),
            "per_key_identity": "shape_dtype_torch_equal",
            "ignored_classification_tensors": len(excluded),
            "ignored_classification_keys": excluded,
            "fusion_bilinear_modules": self.resize_count,
            "timm_version": timm.__version__,
        }
        self.checkpointing = checkpointing
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # Running statistics stay fixed; affine parameters remain trainable.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def _call(self, function, *args):
        if self.checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    def forward(self, image):
        n = self.net
        stem = self._call(lambda x: n.act1(n.bn1(n.conv1(x))), image)
        x = self._call(lambda x: n.act2(n.bn2(n.conv2(x))), stem)
        for block in n.layer1:
            x = self._call(block, x)
        values = [x]
        for transition, stage in ((n.transition1, n.stage2), (n.transition2, n.stage3),
                                   (n.transition3, n.stage4)):
            values = [self._call(layer, values[i] if i < len(values) else values[-1])
                      for i, layer in enumerate(transition)]
            for block in stage:
                # timm mutates list inputs: rebuild the list during recomputation.
                def apply_module(*xs, current=block):
                    return tuple(current(list(xs)))
                values = list(self._call(apply_module, *values))
        return stem, values


class ChannelLayerNorm(nn.Module):
    """Independent C-axis LayerNorm at each N,Y,X; FP32 statistics and affine."""
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = int(channels)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f"ChannelLayerNorm expected N,{self.channels},H,W")
        dtype = x.dtype
        # Combine FP32 conversion and NHWC layout in one allocation. Preserving
        # the NCHW stride here would make layer_norm clone another FP32 grid.
        # Disable autocast explicitly so reductions never use BF16/FP16.
        with torch.autocast(device_type=x.device.type, enabled=False):
            normalized_input = x.movedim(1, -1).to(dtype=torch.float32,
                                                  memory_format=torch.contiguous_format)
            value = F.layer_norm(normalized_input, (self.channels,),
                                 self.weight.float(), self.bias.float(), self.eps)
        return value.movedim(-1, 1).to(dtype)


class ConvLN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, groups=1):
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2,
                                   groups=groups, bias=False),
                         ChannelLayerNorm(out_channels), nn.SiLU())


class Refine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local = ConvLN(channels, channels, 3, channels)
        self.context = ConvLN(channels, channels, 7, channels)
        self.mix = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False),
                                 ChannelLayerNorm(channels))

    def forward(self, x):
        return F.silu(x + self.mix(self.local(x) + self.context(x)))


@torch.no_grad()
def logit_statistics(value, max_samples=65536):
    """Return detached scalar diagnostics without retaining a feature map.

    Mean, RMS and max use all elements. Absolute quantiles use a deterministic
    stride sample bounded to max_samples per class, explicitly labelled in
    the result; set max_samples=None for exact quantiles on small test grids.
    Native direct logits are summarized before their F1-grid interpolation.
    """
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive or None")
    value = value.detach().float()
    mean, rms, abs_max, quantiles, sample_counts = [], [], [], [], []
    elements_per_class = value.shape[0] * value.shape[2] * value.shape[3]
    for channel in value.unbind(1):
        # At most one class is flattened; no full high-resolution copy is kept.
        flat = channel.reshape(-1)
        stride = 1 if max_samples is None else max(1, math.ceil(flat.numel() / max_samples))
        sample = flat[::stride].abs()
        mean.append(flat.mean())
        rms.append(torch.linalg.vector_norm(flat) / math.sqrt(flat.numel()))
        abs_max.append(torch.maximum(flat.amax().abs(), flat.amin().abs()))
        quantiles.append(torch.quantile(sample, sample.new_tensor([.5, .9, .99])))
        sample_counts.append(sample.numel())
    # Convert only scalars to host values; never return references to activations.
    return {
        "mean": torch.stack(mean).tolist(),
        "rms": torch.stack(rms).tolist(),
        "abs_max": torch.stack(abs_max).tolist(),
        "abs_quantile_levels": [.5, .9, .99],
        "abs_quantiles": torch.stack(quantiles).tolist(),
        "elements_per_class": elements_per_class,
        "quantile_samples_per_class": sample_counts,
        "quantile_method": "exact" if all(n == elements_per_class for n in sample_counts)
                           else "deterministic_flat_stride_sample",
    }


class PyramidDecoder(nn.Module):
    def __init__(self, checkpointing=True):
        super().__init__()
        self.checkpointing = checkpointing
        self.deep = nn.Sequential(ConvLN(384, 128), Refine(128))
        self.laterals = nn.ModuleList([ConvLN(192, 128), ConvLN(96, 96), ConvLN(48, 64)])
        self.fusions = nn.ModuleList([nn.Sequential(ConvLN(256, 128), Refine(128)),
                                      nn.Sequential(ConvLN(224, 96), Refine(96)),
                                      nn.Sequential(ConvLN(160, 64), Refine(64))])
        self.stem = ConvLN(64, 32)
        self.detail = nn.Sequential(ConvLN(96, 48), Refine(48), Refine(48))
        # C replaces the old auxiliary head, with its redundant bias removed.
        self.auxiliary = nn.Conv2d(64, 4, 1, bias=False)
        self.readout = nn.Conv2d(48, 16, 1, bias=True)
        self.direct_classifiers = nn.ModuleList([
            nn.Conv2d(channels, 4, 1, bias=False) for channels in (48, 96, 192, 384)])
        self.direct_bias = nn.Parameter(torch.full((4,), math.log(.01 / .99)))
        # Other convolutions retain the uniform Conv2d default initialization;
        # ChannelLayerNorm affine is explicitly gamma=1, beta=0.
        for layer in self.direct_classifiers:
            nn.init.normal_(layer.weight, std=.01)
        nn.init.zeros_(self.auxiliary.weight)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        if sum(p.numel() for p in self.parameters()) != DECODER_PARAMETERS:
            raise RuntimeError("Decoder parameter count differs from the v3 contract")

    def _call(self, function, *args):
        if self.checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    def forward(self, image, stem, features, *, diagnostics=False,
                diagnostic_max_samples=65536):
        f1, f2, f3, f4 = features
        direct = None
        direct_diagnostics = []
        for i, (classifier, feature) in enumerate(zip(self.direct_classifiers, features)):
            contribution = classifier(feature).float()
            if diagnostics:
                direct_diagnostics.append({"scale": f"F{i + 1}", "grid": list(feature.shape[-2:]),
                    **logit_statistics(contribution, None)})
            # Sum on the F1 grid first; do not collapse this and final upsampling.
            projected = resize(contribution, f1.shape[-2:])
            direct = projected if direct is None else direct + projected
        # The design specifies summation first, followed by one shared bias.
        direct = direct + self.direct_bias.float().view(1, 4, 1, 1)
        x = self._call(self.deep, f4)
        for lateral, fusion, feature in zip(self.laterals, self.fusions, (f3, f2, f1)):
            def merge(coarse, fine, project=lateral, refine=fusion):
                return refine(torch.cat((resize(coarse, fine.shape[-2:]), project(fine)), 1))
            x = self._call(merge, x, feature)
        semantic_residual = self.auxiliary(x).float()
        semantic = direct + semantic_residual

        def read_detail(semantic_feature, shallow):
            detail = torch.cat((resize(semantic_feature, shallow.shape[-2:]), self.stem(shallow)), 1)
            return F.pixel_shuffle(self.readout(self.detail(detail)), 2).float()

        detail_residual = self._call(read_detail, x, stem)
        if detail_residual.shape[-2:] != image.shape[-2:]:
            raise ValueError("Stem PixelShuffle output does not match the input canvas")
        logits = resize(semantic, image.shape[-2:]) + detail_residual
        output = {"logits": logits, "auxiliary_logits": semantic}
        if diagnostics:
            output["diagnostics"] = {
                "class_order": ["EX", "HE", "SE", "MA"],
                "direct_logits": direct_diagnostics,
                "semantic_residual": logit_statistics(semantic_residual, diagnostic_max_samples),
                "detail_residual": logit_statistics(detail_residual, diagnostic_max_samples),
            }
        return output


class NativePyramid(nn.Module):
    architecture = ARCHITECTURE_VERSION

    def __init__(self, pretrained, *, checkpointing=True, seed=42, head_seed=20261011,
                 memory_format="contiguous"):
        super().__init__()
        if memory_format not in ("contiguous", "channels_last"):
            raise ValueError("Unsupported convolution memory format")
        self.execution_memory_format = memory_format
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.backbone = W48Features(pretrained, checkpointing)
            torch.random.default_generator.manual_seed(head_seed)
            self.decoder = PyramidDecoder(checkpointing)
        if sum(p.numel() for p in self.parameters()) != MODEL_PARAMETERS:
            raise RuntimeError("Full model parameter count differs from the v3 contract")
        if memory_format == "channels_last":
            self.to(memory_format=torch.channels_last)

    def forward(self, image, *, diagnostics=False, diagnostic_max_samples=65536):
        if image.ndim != 4 or image.shape[1] != 3 or any(v % 32 for v in image.shape[-2:]):
            raise ValueError("Expected BCHW RGB with height/width divisible by 32")
        if image.shape[0] < 1:
            raise ValueError("Expected at least one image")
        # Frozen BN and per-pixel LN make images independent. Use the same
        # per-image convolution schedule for a complete batch and microbatches:
        # cuDNN's B=2/B=1 kernels otherwise perturb logits/OHEM and gradients.
        # The caller still computes the complete B=2 loss/backward when asked;
        # this does not split its loss, denominators or optimizer update.
        if image.shape[0] > 1:
            outputs = [self.forward(item, diagnostics=diagnostics,
                       diagnostic_max_samples=diagnostic_max_samples) for item in image.split(1)]
            result = {key: torch.cat([output[key] for output in outputs], dim=0)
                      for key in ("logits", "auxiliary_logits")}
            if diagnostics:
                result["diagnostics"] = {"per_image": [output["diagnostics"] for output in outputs]}
            return result
        if self.execution_memory_format == "channels_last":
            image = image.contiguous(memory_format=torch.channels_last)
        stem, features = self.backbone(image)
        return self.decoder(image, stem, features, diagnostics=diagnostics,
                            diagnostic_max_samples=diagnostic_max_samples)

    @torch.no_grad()
    def predict(self, image, original_hw):
        return resize(self(image)["logits"].float(), original_hw).sigmoid()

