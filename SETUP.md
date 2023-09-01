# Setup Guide

- create new virtual env

```python -m venv venv```

```. venv/bin/activate```

- install requirements

```pip install -r requirements.txt```

(torch may need)
```pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116```

- install utilities as editable package

```pip install --editable utilities```

you may need to do ```pip install --upgrade pip``` first

- check pc_info for possible discrepancies

- run bash scripts
