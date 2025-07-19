from typing import Tuple, Optional
from dataclasses import dataclass
from lib.utils import BaseConfig


@dataclass
class BiLSTMConfig(BaseConfig):
    # Training
    batch_size: int = 32
    num_epochs: int = 10
    max_grad_norm: float = 5.0

    # Optimizer (Adam) & Linear Scheduler
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    warmup_steps: int = 500

    # Model architecture
    vocab_size: Optional[int] = None  # Required for later
    embedding_dim: int = 256
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.4

    # Misc
    tokenizer_name: str = "bert-base-uncased"
    num_labels: int = 2
    max_seq_length: int = 128
    save_every: int = 1
    seed: int = 42
