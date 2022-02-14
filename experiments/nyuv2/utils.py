import numpy as np
import torch


class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).cpu().numpy(), acc.cpu().numpy()


def depth_error(x_pred, x_output):
    device = x_pred.device
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    return (
        torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)
    ).item(), (
        torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)
    ).item()


def normal_error(x_pred, x_output):
    binary_mask = torch.sum(x_output, dim=1) != 0
    error = (
        torch.acos(
            torch.clamp(
                torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1
            )
        )
        .detach()
        .cpu()
        .numpy()
    )
    error = np.degrees(error)
    return (
        np.mean(error),
        np.median(error),
        np.mean(error < 11.25),
        np.mean(error < 22.5),
        np.mean(error < 30),
    )


# for calculating \Delta_m
delta_stats = [
    "mean iou",
    "pix acc",
    "abs err",
    "rel err",
    "mean",
    "median",
    "<11.25",
    "<22.5",
    "<30",
]
BASE = np.array(
    [0.3830, 0.6376, 0.6754, 0.2780, 25.01, 19.21, 0.3014, 0.5720, 0.6915]
)  # base results from CAGrad
SIGN = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1])
KK = np.ones(9) * -1


def delta_fn(a):
    return (KK ** SIGN * (a - BASE) / BASE).mean() * 100.0  # * 100 for percentage
