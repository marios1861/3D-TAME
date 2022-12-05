from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    with open(cfg_path.parents[0] / "default.yaml") as f:
        defaults = yaml.safe_load(f)
    defaults.update(cfg)
    return defaults
