from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(cfg: str) -> Dict[str, Any]:
    ROOT_DIR = get_project_root()
    cfg_path = ROOT_DIR / "configs" / f"{cfg}.yaml"
    with open(cfg_path) as f:
        cfg_dict: Dict[str, Any] = yaml.safe_load(f)
    with open(cfg_path.parents[0] / "default.yaml") as f:
        defaults: Dict[str, Any] = yaml.safe_load(f)
    defaults.update(cfg_dict)
    return defaults


def get_project_root() -> Path:
    return Path(__file__).parents[3]
