import matplotlib
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from experiments.toy.problem import Toy


# plotting utils
def plot_2d_pareto(trajectories: dict, scale):
    """Adaptation of code from: https://github.com/Cranial-XIX/CAGrad"""
    fig, ax = plt.subplots(figsize=(6, 5))

    F = Toy(scale=scale)

    losses = []
    for res in trajectories.values():
        losses.append(F.batch_forward(torch.from_numpy(res["traj"])))

    yy = -8.3552
    x = np.linspace(-7, 7, 1000)

    inpt = np.stack((x, [yy] * len(x))).T
    Xs = torch.from_numpy(inpt).double()

    Ys = F.batch_forward(Xs)
    ax.plot(
        Ys.numpy()[:, 0],
        Ys.numpy()[:, 1],
        "-",
        linewidth=8,
        color="#72727A",
        label="Pareto Front",
    )  # Pareto front

    for i, tt in enumerate(losses):
        ax.scatter(
            tt[0, 0],
            tt[0, 1],
            color="k",
            s=150,
            zorder=10,
            label="Initial Point" if i == 0 else None,
        )
        colors = matplotlib.cm.magma_r(np.linspace(0.1, 0.6, tt.shape[0]))
        ax.scatter(tt[:, 0], tt[:, 1], color=colors, s=5, zorder=9)

    sns.despine()
    ax.set_xlabel(r"$\ell_1$", size=30)
    ax.set_ylabel(r"$\ell_2$", size=30)
    ax.xaxis.set_label_coords(1.015, -0.03)
    ax.yaxis.set_label_coords(-0.01, 1.01)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    plt.tight_layout()

    legend = ax.legend(
        loc=2, bbox_to_anchor=(-0.15, 1.3), frameon=False, fontsize=20, ncol=2
    )

    return ax, fig, legend
