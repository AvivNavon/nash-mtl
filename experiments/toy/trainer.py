import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from experiments.toy.utils import plot_2d_pareto
from experiments.utils import (
    common_parser,
    extract_weight_method_parameters_from_args,
    set_logger,
)
from experiments.toy.problem import Toy
from methods.weight_methods import WeightMethods

set_logger()


def main(method_type, device, n_iter, scale):
    weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    n_tasks = 2

    F = Toy(scale=scale)

    all_traj = dict()

    # the initial positions
    inits = [
        torch.Tensor([-8.5, 7.5]),
        torch.Tensor([0.0, 0.0]),
        torch.Tensor([9.0, 9.0]),
        torch.Tensor([-7.5, -0.5]),
        torch.Tensor([9, -1.0]),
    ]

    for i, init in enumerate(inits):
        traj = []
        x = init.clone()
        x.requires_grad = True
        x = x.to(device)

        method = WeightMethods(
            method=method_type,
            device=device,
            n_tasks=n_tasks,
            **weight_methods_parameters[method_type],
        )

        optimizer = torch.optim.Adam(
            [
                dict(params=[x], lr=1e-3),
                dict(params=method.parameters(), lr=args.method_params_lr),
            ],
        )

        for _ in tqdm(range(n_iter)):
            traj.append(x.cpu().detach().numpy().copy())

            optimizer.zero_grad()
            f = F(x, False)
            _ = method.backward(
                losses=f,
                shared_parameters=(x, ),
                task_specific_parameters=None,
                last_shared_parameters=None,
                representation=None,
            )
            optimizer.step()

        all_traj[i] = dict(init=init.cpu().detach().numpy().copy(), traj=np.array(traj))

    return all_traj


if __name__ == "__main__":
    parser = ArgumentParser("Toy example (modification of the one in CAGrad)", parents=[common_parser])
    parser.set_defaults(
        n_epochs=35000,
        method="nashmtl",
        data_path=None
    )
    parser.add_argument("--scale", default=1e-1, type=float, help="scale for first loss")
    parser.add_argument("--out-path", default="outputs", type=Path, help="output path")
    args = parser.parse_args()

    out_path = args.out_path
    out_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Logs and plots are saved in: {out_path.as_posix()}")

    device = torch.device("cpu")
    all_traj = main(
        method_type=args.method,
        device=device,
        n_iter=args.n_epochs,
        scale=args.scale
    )

    # plot
    ax, fig, legend = plot_2d_pareto(trajectories=all_traj, scale=args.scale)

    title_map = {'nashmtl': 'Nash-MTL', 'cagrad': 'CAGrad', 'mgda': 'MGDA', 'pcgrad': 'PCGrad', 'ls': 'LS'}
    ax.set_title(title_map[args.method], fontsize=25)
    plt.savefig(out_path / f"{args.method}.png", bbox_extra_artists=(legend, ), bbox_inches='tight', facecolor='white')
