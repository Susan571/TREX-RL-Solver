"""
TREX (Topologically-guided Reinforcement Learning with EXploration) module.

This module implements the TREX algorithm for AC problem solving, including:
- ASC: Active Symbolic Closure (validity masking)
- TNSC: Topological Neuro-Symbolic Compression (potentials)
- Topologically Guided Sampling
- Hybrid Reward shaping
- TREX-style PPO updates
"""

from trex.policy import TREXPolicy
from trex.train import train_trex

__all__ = ["TREXPolicy", "train_trex"]
