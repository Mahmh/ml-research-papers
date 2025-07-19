from typing import Tuple, Optional
from dataclasses import dataclass
from lib.utils import BaseConfig


@dataclass
class LogisticRegressionConfig(BaseConfig):
    # Training
    batch_size: int = 32
    num_epochs: int = 10

    # Optimizer (SGD)
    learning_rate: float = 1.0
    weight_decay: float = 0.0

    # Tokenizer (TF-IDF)
    max_features: Optional[int] = None
    ngram_range: Tuple[int, int] = (1, 1)

    # Misc
    input_dim: Optional[int] = None  # Required for later
    num_labels: int = 2
    save_every: int = 3
    seed: int = 42
