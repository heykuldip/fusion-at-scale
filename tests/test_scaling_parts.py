"""The scaling benchmark's parts on CPU: the synthetic data are fixed by the seed, and the model maps a batch to class
scores."""
import torch

from fusion_scale.data.synthetic_features import SyntheticFeatures
from fusion_scale.models.features_model import FeaturesModel


def test_synthetic_features_are_fixed_by_the_seed():
    a, b = SyntheticFeatures(n=64, seed=1), SyntheticFeatures(n=64, seed=1)
    assert torch.equal(a.x, b.x) and torch.equal(a.y, b.y)


def test_model_maps_a_batch_to_class_scores():
    x, y = SyntheticFeatures(n=8, seed=0)[0:8]
    assert FeaturesModel(width=32, depth=1, heads=4)(x).shape == (8, 13)
