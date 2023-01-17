# T-TAME: Trainable Attention Mechanism for Explanations, now for transformers

## readme for dev purposes only

- [T-TAME: Trainable Attention Mechanism for Explanations, now for transformers](#t-tame-trainable-attention-mechanism-for-explanations-now-for-transformers)
  - [readme for dev purposes only](#readme-for-dev-purposes-only)
  - [Initial Setup](#initial-setup)
  - [Available scripts](#available-scripts)

## Initial Setup

Make sure that you have a working git, cuda, Python 3 and pip installation before proceeding.

- Clone this repository:

```commandline
git clone https://github.com/bmezaris/TAME
```

- Go to the locally saved repository path:

```commandline
cd TAME
```

- Create and activate a virtual environment:

```commandline
python3 -m venv ./venv
. ./venv/bin/activate
```

- Install project requirements:

```commandline
pip install -r requirements.txt
```

> __Note__: you may have to install the libraries torch and torchvision separately. Follow the pip instructions [here](https://pytorch.org/get-started/locally/). Also, make sure you have a working matplotlib GUI backend (such as qt with `pip install PyQt6`).

## Available scripts

You can evaluate the attention mechanism trained on vit on the 7th epoch
 using the command:

```bash
tame val ---cfg vit_b_16.yaml epoch 7
```
