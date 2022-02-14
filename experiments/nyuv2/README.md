# NYUv2 Experiment

Modification of the code in [CAGrad](https://github.com/Cranial-XIX/CAGrad) and [MTAN](https://github.com/lorenmt/mtan).

## Dataset

The dataset is available at [this link](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0). Put the downloaded files in `./dataset` so that the folder structure is `.dataset/train` and `./dataset/val`.   

## Evaluation

To align with previous work on MTL [Liu et al. (2019)](https://arxiv.org/abs/1803.10704); [Yu et al. (2020)](https://arxiv.org/abs/2001.06782); [Liu et al. (2021)](https://arxiv.org/pdf/2110.14048.pdf) we report the test performance averaged
over the last 10 epochs. Note that this averaging is not handled in the code and need to be applied by the user. 
