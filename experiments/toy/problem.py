import torch
from torch import nn

LOWER = 0.000005


class Toy(nn.Module):
    def __init__(self, scale=1.0, scale_both_losses=1.0):
        super(Toy, self).__init__()
        self.centers = torch.Tensor([[-3.0, 0], [3.0, 0]])
        self.scale = scale
        self.scale_both_losses = scale_both_losses

    def forward(self, x, compute_grad=False):
        x1 = x[0]
        x2 = x[1]

        f1 = torch.clamp((0.5 * (-x1 - 7) - torch.tanh(-x2)).abs(), LOWER).log() + 6
        f2 = torch.clamp((0.5 * (-x1 + 3) + torch.tanh(-x2) + 2).abs(), LOWER).log() + 6
        c1 = torch.clamp(torch.tanh(x2 * 0.5), 0)

        f1_sq = ((-x1 + 7).pow(2) + 0.1 * (-x2 - 8).pow(2)) / 10 - 20
        f2_sq = ((-x1 - 7).pow(2) + 0.1 * (-x2 - 8).pow(2)) / 10 - 20
        c2 = torch.clamp(torch.tanh(-x2 * 0.5), 0)

        f1 = f1 * c1 + f1_sq * c2
        f1 *= self.scale
        f2 = f2 * c1 + f2_sq * c2

        f = torch.stack([f1, f2]) * self.scale_both_losses
        if compute_grad:
            g11 = torch.autograd.grad(f1, x1, retain_graph=True)[0].item()
            g12 = torch.autograd.grad(f1, x2, retain_graph=True)[0].item()
            g21 = torch.autograd.grad(f2, x1, retain_graph=True)[0].item()
            g22 = torch.autograd.grad(f2, x2, retain_graph=True)[0].item()
            g = torch.Tensor([[g11, g21], [g12, g22]])
            return f, g
        else:
            return f

    def batch_forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]

        f1 = torch.clamp((0.5 * (-x1 - 7) - torch.tanh(-x2)).abs(), LOWER).log() + 6
        f2 = torch.clamp((0.5 * (-x1 + 3) + torch.tanh(-x2) + 2).abs(), LOWER).log() + 6
        c1 = torch.clamp(torch.tanh(x2 * 0.5), 0)

        f1_sq = ((-x1 + 7).pow(2) + 0.1 * (-x2 - 8).pow(2)) / 10 - 20
        f2_sq = ((-x1 - 7).pow(2) + 0.1 * (-x2 - 8).pow(2)) / 10 - 20
        c2 = torch.clamp(torch.tanh(-x2 * 0.5), 0)

        f1 = f1 * c1 + f1_sq * c2
        f1 *= self.scale
        f2 = f2 * c1 + f2_sq * c2

        f = torch.cat([f1.view(-1, 1), f2.view(-1, 1)], -1) * self.scale_both_losses
        return f
