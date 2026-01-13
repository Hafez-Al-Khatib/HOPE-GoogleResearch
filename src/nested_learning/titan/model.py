from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn 

from ..backbones import AttentionConfig, SelfAttention
from ..levels import LevelSpec 
from ..optim.manager import LevelConfig, LevelOptimizerManager
from ..titan.memory import TitanMemory, TitanMemoryConfig
from ..hope.self_mod import SelfModifier


@dataclass 
class TitanOnlyModelConfig:
    vocab_size: int
    dim: int
    num_layers: int
    heads: int
    