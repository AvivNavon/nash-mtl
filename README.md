# Nash-MTL

Official implementation of _"Multi-Task Learning as a Bargaining Game"_.

<p align="center"> 
    <img src="https://github.com/AvivNavon/nash-mtl/blob/main/misc/toy_pareto_2d.png" width="800">
</p>

## Setup environment

```bash
conda create -n nashmtl python=3.9.7
conda activate nashmtl
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=10.2 -c pytorch
conda install pyg -c pyg -c conda-forge
```

Install the repo:

```bash
git clone https://github.com/AvivNavon/nash-mtl.git
cd nash-mtl
pip install -e .
```

## Run experiment

To run experiments:

```bash
cd experiment/<experiment name>
python trainer.py --method=nashmtl
```
Follow instruction on the experiment README file for more information regarding, e.g., datasets.  

Here `<experiment name>` is one of `[toy, quantum_chemistry, nyuv2]`. You can also replace `nashmtl` with on of the following MTL methods.

We also support experiment tracking with **[Weights & Biases](https://wandb.ai/site)** with two additional parameters:

```bash
python trainer.py --method=nashmtl --wandb_project=<project-name> --wandb_entity=<entity-name>
```

## MTL methods

We support the following MTL methods with a unified API. To run experiment with MTL method `X` simply run:
```bash
python trainer.py --method=X
```

| Method (code name) | Paper (notes) |
| :---: | :---: |
| Nash-MTL (`nashmtl`) | [Multi-Task Learning as a Bargaining Game](https://arxiv.org/pdf/2202.01017v1.pdf) |
| CAGrad (`cagrad`) | [Conflict-Averse Gradient Descent for Multi-task Learning](https://arxiv.org/pdf/2110.14048.pdf) |
| PCGrad (`pcgrad`) | [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782) |
| IMTL-G (`imtl`) | [Towards Impartial Multi-task Learning](https://openreview.net/forum?id=IMPnRXEWpvr) |
| MGDA (`mgda`) | [Multi-Task Learning as Multi-Objective Optimization](https://arxiv.org/abs/1810.04650) |
| DWA (`dwa`) | [End-to-End Multi-Task Learning with Attention](https://arxiv.org/abs/1803.10704) |
| Uncertainty weighting (`uw`) | [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/pdf/1705.07115v3.pdf) |
| Linear scalarization (`ls`) | - (equal weighting) |
| Scale-invariant baseline (`scaleinvls`) | - (see Nash-MTL paper for details) |
| Random Loss Weighting (`rlw`) | [A Closer Look at Loss Weighting in Multi-Task Learning](https://arxiv.org/pdf/2111.10603.pdf) |




## Citation

If you find `Nash-MTL` to be useful in your own research, please consider citing the following paper:

```bib
@article{navon2022multi,
  title={Multi-Task Learning as a Bargaining Game},
  author={Navon, Aviv and Shamsian, Aviv and Achituve, Idan and Maron, Haggai and Kawaguchi, Kenji and Chechik, Gal and Fetaya, Ethan},
  journal={arXiv preprint arXiv:2202.01017},
  year={2022}
}
```
