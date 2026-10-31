"""Class routing fixed solely from E4/B validation class IoUs; no threshold search."""
from copy import deepcopy
import torch
from mmcv.runner import load_checkpoint
from mmseg.models.builder import SEGMENTORS
from hr100_refinement import HR100Refinement


@SEGMENTORS.register_module()
class HR100ClasswiseEnsemble(HR100Refinement):
    def __init__(self, reference_checkpoint, primary_classes=(0, 2), **kwargs):
        reference_kwargs = deepcopy(kwargs)
        super().__init__(**kwargs)
        assert self.ma_refiner is None
        self.reference = HR100Refinement(**reference_kwargs)
        load_checkpoint(self.reference, reference_checkpoint, map_location='cpu', strict=True)
        self.reference.requires_grad_(False)
        self.primary_classes = tuple(primary_classes)

    def encode_decode(self, img, img_metas):
        primary = super().encode_decode(img, img_metas)
        reference = self.reference.encode_decode(img, img_metas)
        mask = torch.tensor([c in self.primary_classes for c in range(4)],
                            device=primary.device).view(1, 4, 1, 1)
        return torch.where(mask, primary, reference)

    def forward_train(self, *args, **kwargs):
        raise RuntimeError('Inference-only, validation-selected two-model ensemble')


@SEGMENTORS.register_module()
class HR100MeanEnsemble(HR100ClasswiseEnsemble):
    def encode_decode(self, img, img_metas):
        return HR100Refinement.encode_decode(self, img, img_metas)

    def inference(self, img, img_meta, rescale):
        # Average probabilities after each model's original-size interpolation
        # and sigmoid, with identical 0.5 weight for every pixel and class.
        assert self.use_sigmoid
        primary = super().inference(img, img_meta, rescale)
        reference = self.reference.inference(img, img_meta, rescale)
        return (primary + reference) * 0.5
