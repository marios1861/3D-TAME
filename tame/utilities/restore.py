import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from torch.optim import Optimizer

from .composite_models import Generic


def load_model(
    cfg_name: str,
    cfg: Dict[str, Any],
    model: Generic,
    optimizer: Optional[Optimizer] = None,
    epoch: Optional[int] = None
) -> int:
    checkpoint_dir = get_checkpoint_dir(cfg_name, cfg, epoch)

    if checkpoint_dir.is_file():
        print(f'Loading checkpoint from file {checkpoint_dir}')
        checkpoint = torch.load(checkpoint_dir)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        model.attn_mech.load_state_dict(checkpoint_dir['attn_mech'])
        return checkpoint['epoch']
    else:
        return 0


def save_model(
    cfg_name: str,
    cfg: Dict[str, Any],
    model: Generic,
    optimizer: Optimizer,
    epoch: int,
):
    checkpoint_dir = get_checkpoint_dir(cfg_name, cfg, epoch)
    torch.save({'attn_mech': model.attn_mech.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch}, checkpoint_dir)


def get_checkpoint_dir(cfg_name: str,
                       cfg: Dict[str, Any],
                       epoch: Optional[int] = None) -> Path:
    cfg_name = Path(cfg_name).stem  # remove file extension:
    # we have already read the snapshot_ids.csv file
    if 'checkpoint_dir' in cfg.keys():
        checkpoint_dir = cfg["checkpoint_dir"]
    else:
        id_file = Path(cfg['snapshot_dir']) / "snapshot_ids.csv"
        # we are reading the snapshot_ids.csv file for the first time
        if id_file.is_file():
            df = pd.read_csv(id_file)
            # the id for this cfg file already exists
            if cfg_name in df['name']:
                id = df[df['name'] == cfg_name]['id']
            # the id for this cfg file does not exist
            # generate it and update the snapshot_ids.csv file
            else:
                id = uuid.uuid4().hex
                new_id = {"id": id, "name": cfg_name}
                df.append(new_id, ignore_index=True)
                df.to_csv(id_file, mode='a', header=False)
        # the id for this cfg does not exist
        # the snapshot_ids.csv file does not exist
        # generate id and create file
        else:
            id = uuid.uuid4().hex
            new_id = {"id": id, "name": cfg_name}
            df.to_csv(id_file, mode='w', header=True)
    checkpoint_dir: Path = Path(cfg['snapshot_dir']) / id
    cfg["checkpoint_dir"] = checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if epoch:
        checkpoint_dir = checkpoint_dir / f'epoch_{epoch}.pt'
    else:
        epochs = [int(x.stem.replace("epoch", "")) for x in checkpoint_dir.iterdir()]
        checkpoint_dir = checkpoint_dir / f'epoch_{max(epochs)}.pt'
    return checkpoint_dir
