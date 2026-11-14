import unittest
from torch import nn
from scripts.round2_experiments import set_phase, checkpoint_score


class Round2Checks(unittest.TestCase):
    def test_frozen_batchnorm_and_training_phases(self):
        model = nn.Module()
        model.early = nn.Sequential(nn.Conv2d(2,2,1),nn.BatchNorm2d(2))
        model.layer4 = nn.Sequential(nn.Conv2d(2,2,1),nn.BatchNorm2d(2))
        model.fc = nn.Linear(2,1)
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.fc.parameters():
            p.requires_grad_(True)
        set_phase(model, True)
        self.assertTrue(model.fc.weight.requires_grad)
        self.assertFalse(model.layer4[0].weight.requires_grad)
        set_phase(model, False)
        self.assertTrue(model.layer4[0].weight.requires_grad)
        self.assertFalse(model.early[0].weight.requires_grad)
        for bn in [model.early[1],model.layer4[1]]:
            self.assertFalse(bn.training)
            self.assertFalse(bn.weight.requires_grad)

    def test_selection_prioritizes_false_positive_control(self):
        a = checkpoint_score({'specificity':.8,'auroc':.85})
        b = checkpoint_score({'specificity':.7,'auroc':.9})
        self.assertGreater(a,b)


if __name__ == '__main__':
    unittest.main()
