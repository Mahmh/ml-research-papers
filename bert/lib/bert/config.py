from typing import Tuple
from dataclasses import dataclass
from lib.utils import BaseConfig


@dataclass
class BERTConfig(BaseConfig):
    # Training
    batch_size: int = 32
    num_epochs: int = 3
    max_grad_norm: float = 1.0

    # Optimizer (AdamW) & Linear Scheduler
    learning_rate: float = 2e-5
    betas: Tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-6
    weight_decay: float = 1e-2
    warmup_steps: int = 0

    # Misc
    num_labels: int = 2
    max_seq_length: int = 128
    save_every: int = 3
    seed: int = 42
    pretrained_model_name: str = "bert-base-uncased"
    dropout_prob: float = 0.1
