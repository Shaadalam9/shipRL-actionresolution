"""Project-wide settings read from the JSON ``config`` file in the repo root.

These are presentation and run-selection choices that are not part of an
experiment definition, so they live outside the per-run YAML configs: which
controllers to compare, the figure font, the Plotly template, and the log
level. ``config`` is read if it exists and ``default.config`` is used as the
fallback, matching :func:`common.get_configs`; a key missing from both falls
back to the value baked in here.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIG_FILE = REPO_ROOT / "config"
DEFAULT_CONFIG_FILE = REPO_ROOT / "default.config"

#: Used when a key is absent from both config files.
FALLBACKS: dict[str, Any] = {
    "font_family": "Helvetica, Arial, sans-serif",
    "font_size": 12,
    "plotly_template": "plotly_white",
    "logger_level": "info",
    "decimal_places": 2,
    "agents": [],
}


@lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    """Read the config file once, falling back to ``default.config``."""
    for path in (CONFIG_FILE, DEFAULT_CONFIG_FILE):
        try:
            with open(path) as fid:
                return json.load(fid)
        except FileNotFoundError:
            continue
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: config file is not valid JSON: {exc}") from exc
    return {}


def get(entry_name: str, default: Any = None) -> Any:
    """Return one config entry, or its fallback when it is not configured."""
    if default is None:
        default = FALLBACKS.get(entry_name)
    return load().get(entry_name, default)


#: Third-party loggers held at WARNING regardless of ``logger_level``. Kaleido
#: and its browser driver narrate every image export at INFO, which would bury
#: the per-controller results the scripts print.
_NOISY_MODULES = (
    "absl",
    "choreographer",
    "h11",
    "kaleido",
    "logistro",
    "matplotlib",
    "PIL",
    "requests",
    "tensorflow",
    "urllib3",
)


def configure_logging(program_name: str = "asv_rl") -> None:
    """Set the root log level from the ``logger_level`` config entry."""
    level = str(get("logger_level")).upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s - %(levelname)-8s - %(name)s - %(message)s",
    )
    for module in _NOISY_MODULES:
        logging.getLogger(module).setLevel(logging.WARNING)
    logging.getLogger(program_name).debug("Logging configured at level %s", level)
