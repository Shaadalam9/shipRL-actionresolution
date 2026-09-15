"""Agent definitions and the registry that maps a config name to one."""

from asv_rl.agents.registry import ALGORITHM_NAMES, AlgorithmSpec, get_spec, load_config

__all__ = ["ALGORITHM_NAMES", "AlgorithmSpec", "get_spec", "load_config"]
