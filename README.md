# Nash-MTL

Official implementation of _"Multi-Task Learning as a Bargaining Game"_.

## Setup environment

```bash
conda create -n nashmtl python=3.9.7
conda activate nashmtl
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
conda install pyg -c pyg -c conda-forge
```

Next -> `cd` to the unzipped folder i.e. `cd <UNZIPPED FOLDER>`
RUN -> `pip install -e .`

## QM9 Experiment 

To run Nash-MTL:

```bash
cd experiments/quantum_chemistry
python trainer.py --method=nashmtl
```

To train using another MTL method simply replace `'nashmtl'` with one of `['cagrad', 'pcgrad', 'mgda', 'ls', 'uw', 'scaleinvls']`.