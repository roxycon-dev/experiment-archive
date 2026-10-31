"""Single registered intervention; parameters, forward values and direct base path stay unchanged."""
from model import SACDPrivate


class FeatureIsolatedPrivate(SACDPrivate):
    def forward(self, image, features, z0):
        return super().forward(image, tuple(value.detach() for value in features), z0)


def apply_intervention(model, arm):
    assert arm in ('control', 'feature_isolated')
    assert type(model.private) is SACDPrivate
    if arm == 'feature_isolated':
        # No new constructor, RNG consumption, parameter replacement or optimizer remapping.
        model.private.__class__ = FeatureIsolatedPrivate
    return model
