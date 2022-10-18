from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(cfg_path: Path, cfg_name: str) -> Dict[str, Any]:
    with open(cfg_path / cfg_name) as f:
        cfg = yaml.safe_load(f)
    with open(cfg_path / "default.yaml") as f:
        defaults = yaml.safe_load(f)
    defaults.update(cfg)
    return defaults
