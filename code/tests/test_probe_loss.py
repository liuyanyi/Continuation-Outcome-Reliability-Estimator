"""Loss scaling must retain the absolute sqrt(N) weight at batch size one."""
import math
import unittest

import torch

from cognitive_probe.probe_model.probe_model import ForProbeLoss


class ProbeLossTest(unittest.TestCase):
    def test_sqrt_weight_preserves_absolute_scale_and_gradient(self):
        logits = torch.tensor([[0.0]], requires_grad=True)
        loss = ForProbeLoss(logits, labels=torch.tensor([1.0]),
                            label_weights=torch.tensor([4.0]), loss_weight_mode='sqrt_n')
        self.assertAlmostEqual(loss.item(), 2 * math.log(2), places=6)
        loss.backward()
        self.assertAlmostEqual(logits.grad.item(), 0.5)

    def test_supported_weight_modes(self):
        for mode, scale in [('none', 1), ('sqrt_n', 2), ('n', 4)]:
            with self.subTest(mode=mode):
                loss = ForProbeLoss(torch.tensor([[0.0]]), labels=torch.tensor([2.0]),
                                    label_weights=torch.tensor([4.0]), loss_weight_mode=mode)
                self.assertAlmostEqual(loss.item(), scale * math.log(2), places=6)

    def test_removed_weight_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown weight_mode'):
            ForProbeLoss(torch.tensor([[0.0]]), labels=torch.tensor([1.0]),
                         label_weights=torch.tensor([4.0]), loss_weight_mode='sqrt_n_normalized')


if __name__ == '__main__':
    unittest.main()
