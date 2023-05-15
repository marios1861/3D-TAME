# T-TAME: Trainable Attention Mechanism for Explanations, now for transformers

## readme for dev purposes only

- [T-TAME: Trainable Attention Mechanism for Explanations, now for transformers](#t-tame-trainable-attention-mechanism-for-explanations-now-for-transformers)
  - [readme for dev purposes only](#readme-for-dev-purposes-only)
  - [Initial Setup](#initial-setup)
  - [Available scripts](#available-scripts)

## Initial Setup

Make sure that you have a working git, Python 3 and poetry installation before proceeding.

- Clone this repository:

```commandline
git clone git@github.com:marios1861/T-TAME.git
```

- Go to the locally saved repository path:

```commandline
cd T-TAME
```

- Install:

poetry install

> __Note__: You may need to modify the venv activate script in the case that cuda is already installed in your machine. If so, add this line:
> `export LD_LIBRARY_PATH=.../venv/lib/python3.8/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH`

## Available scripts

You can evaluate the attention mechanism trained on vit on the 7th epoch
 using the command:

```bash
tame val ---cfg vit_b_16.yaml epoch 7
```
