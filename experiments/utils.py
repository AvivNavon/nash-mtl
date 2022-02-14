import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from methods import METHODS


def str_to_list(string):
    return [float(s) for s in string.split(",")]


def str_or_float(value):
    try:
        return float(value)
    except:
        return value


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


common_parser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument("--data-path", type=Path, help="path to data")
common_parser.add_argument("--n-epochs", type=int, default=300)
common_parser.add_argument("--batch-size", type=int, default=120, help="batch size")
common_parser.add_argument(
    "--method", type=str, choices=list(METHODS.keys()), help="MTL weight method"
)
common_parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
common_parser.add_argument(
    "--method-params-lr",
    type=float,
    default=0.025,
    help="lr for weight method params. If None, set to args.lr. For uncertainty weighting",
)
common_parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
common_parser.add_argument("--seed", type=int, default=42, help="seed value")
# NashMTL
common_parser.add_argument(
    "--nashmtl-optim-niter", type=int, default=20, help="number of CCCP iterations"
)
common_parser.add_argument(
    "--update-weights-every",
    type=int,
    default=1,
    help="update task weights every x iterations.",
)
# stl
common_parser.add_argument(
    "--main-task",
    type=int,
    default=0,
    help="main task for stl. Ignored if method != stl",
)
# cagrad
common_parser.add_argument("--c", type=float, default=0.4, help="c for CAGrad alg.")
# dwa
# dwa
common_parser.add_argument(
    "--dwa-temp",
    type=float,
    default=2.0,
    help="Temperature hyper-parameter for DWA. Default to 2 like in the original paper.",
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(no_cuda=False, gpus="0"):
    return torch.device(
        f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu"
    )


def extract_weight_method_parameters_from_args(args):
    weight_methods_parameters = defaultdict(dict)
    weight_methods_parameters.update(
        dict(
            nashmtl=dict(
                update_weights_every=args.update_weights_every,
                optim_niter=args.nashmtl_optim_niter,
            ),
            stl=dict(main_task=args.main_task),
            cagrad=dict(c=args.c),
            dwa=dict(temp=args.dwa_temp),
        )
    )
    return weight_methods_parameters
