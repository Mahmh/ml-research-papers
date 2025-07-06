from typing import Tuple
from dataclasses import dataclass
import os

MAX_NUM_EXAMPLES = int(10e5)
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SPECIAL_TOKENS = ['<unk>', '<pad>', '<bos>', '<eos>']

TRAIN_SPLIT = 0.8
VAL_SPLIT   = 0.1
TEST_SPLIT  = 0.1
assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-6, "Splits must sum to 1.0"

@dataclass
class Config:
    input_seq_len: int
    output_seq_len: int
    src_vocab_size: int
    tgt_vocab_size: int
    batch_size: int
    d_model: int
    d_ff: int
    num_heads: int
    num_encoders: int
    num_decoders: int
    p_drop: float
    lr: float
    warmup_steps: int
    num_epochs: int
    save_every: int
    optimizer_betas: Tuple[float, float]
    optimizer_eps: float
    early_stopping_patience: int
    early_stopping_min_delta: float
    src_pad_idx: int
    tgt_pad_idx: int
    tgt_bos_idx: int
    tgt_eos_idx: int