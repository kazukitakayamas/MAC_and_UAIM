"""CPU checks for the switchable MAC score / random control.

    python -m unittest test_mac_score -v

- h0 (default) selects exactly the same pairs as the original select_low_loss_indices,
  and the MeanFlow loss with MAC is bit-identical to the original code path.
- h1 equals the distance between the model's own 1-NFE sample and the paired data.
- random selection uses its own generator and never advances the global RNG
  (so the training randomness is the same as with MAC off / h0).
"""
import unittest

import torch
from torch import nn

from wrappers.utils import (select_low_loss_indices, select_indices_meanflow,
                            meanflow_pair_scores, _rank01)
from wrappers.meanflow import MACWrapper


class TinyNet(nn.Module):
    """model(z, t, h, labels) -> velocity with the same shape as z."""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.3))
        self.b = nn.Parameter(torch.tensor(0.2))
        self.c = nn.Parameter(torch.tensor(0.5))

    def forward(self, z, t, h, labels=None):
        t = t.view(-1, 1, 1, 1)
        h = h.view(-1, 1, 1, 1)
        return self.a * z + self.b * t + self.c * h * torch.tanh(z)


def batch(seed=0, n=8):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 4, 4, generator=g)
    z0 = torch.randn(n, 3, 4, 4, generator=g)
    c = torch.zeros(n, dtype=torch.long)
    return x, z0, c


class ScoreTests(unittest.TestCase):
    def test_h0_is_original_mac(self):
        net = TinyNet()
        x, z0, c = batch()
        old = select_low_loss_indices(net, (x, c), z0, 0.5, model='meanflow')
        new = select_indices_meanflow(net, (x, c), z0, 0.5, score='h0')
        self.assertEqual(sorted(old.tolist()), sorted(new.tolist()))

    def test_h1_is_distance_of_1nfe_sample(self):
        net = TinyNet()
        x, z0, c = batch(1)
        wrapper = MACWrapper(net, None, 1.0, model_type='select')
        x_hat = wrapper.sample(z0, None, None, sample_steps=1)[-1]   # the real 1-NFE sampler
        s1 = meanflow_pair_scores(net, (x, c), z0, score='h1')
        torch.testing.assert_close(s1, ((x_hat - x) ** 2).mean(dim=(1, 2, 3)))

    def test_mix_is_rank_average(self):
        net = TinyNet()
        x, z0, c = batch(2)
        s0 = meanflow_pair_scores(net, (x, c), z0, score='h0')
        s1 = meanflow_pair_scores(net, (x, c), z0, score='h1')
        mix = meanflow_pair_scores(net, (x, c), z0, score='mix')
        torch.testing.assert_close(mix, 0.5 * (_rank01(s0) + _rank01(s1)))

    def test_random_does_not_touch_global_rng(self):
        net = TinyNet()
        x, z0, c = batch(3)
        torch.manual_seed(7)
        before = torch.get_rng_state().clone()
        idx = select_indices_meanflow(net, (x, c), z0, 0.5, selection='random',
                                      generator=torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(len(set(idx.tolist())), 4)


class WrapperTests(unittest.TestCase):
    def make(self, **kw):
        torch.manual_seed(0)
        return MACWrapper(TinyNet(), None, 1.0, model_type='select', **kw)

    def test_h0_loss_identical_to_original_path(self):
        x, _, c = batch(4)
        mac = self.make()                                   # defaults: h0 / model
        torch.manual_seed(11)
        new = mac.forward(x, c, 0.5, global_step=10, total_steps=100, mac_enabled=True)
        torch.manual_seed(11)                               # original MACWrapper.forward
        z0 = torch.randn_like(x)
        idx = select_low_loss_indices(mac.ema_model, (x, c), z0, 0.5, model='meanflow')
        old = mac.get_loss(x, z0, c, indices=idx, global_step=10, total_steps=100)
        self.assertTrue(torch.equal(new, old))

    def test_rng_stream_same_for_off_h0_h1_random(self):
        x, _, c = batch(5)
        states = {}
        for name, kw, enabled in [("off", {}, False), ("h0", {}, True),
                                  ("h1", {"mac_score": "h1"}, True),
                                  ("random", {"mac_selection": "random"}, True)]:
            mac = self.make(**kw)
            torch.manual_seed(3)
            mac.forward(x, c, 0.5, global_step=1, total_steps=10, mac_enabled=enabled)
            states[name] = torch.get_rng_state()
        for name in ("h0", "h1", "random"):
            self.assertTrue(torch.equal(states["off"], states[name]), name)


if __name__ == "__main__":
    unittest.main()
