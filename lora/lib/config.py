from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any


@dataclass
class Config:
    """
    Configuration for fine-tuning with/without LoRA.

    Model/Dataset:
        model_name: HF model id or local path.
        dataset_name: HF dataset id or local path.
        dataset_dir: folder to save & load the dataset.
        train_split: name of the training split.
        val_split: name of the validation split (auto-split if missing).
        val_fraction (float): fraction of the original train split to allocate to the validation set.
        test_split: name of the test split (optional; auto-split if missing).
        test_fraction (float): fraction of the original train split to allocate to the test set.
        text_field: dataset column with text/examples.
        max_seq_length: tokenizer truncation length.

    LoRA:
        use_lora: whether to apply LoRA adapters.
        lora_r, lora_alpha, lora_dropout: LoRA hyperparams.
        target_modules: module names to attach LoRA to.

    Training:
        output_dir: base outputs (plots, artifacts).
        epochs: number of epochs.
        batch_size: per-device batch size.
        grad_accum_steps: steps to accumulate grads.
        lr: learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: LR warmup steps.
        max_grad_norm: gradient clipping value.
        fp16: use autocast FP16/mixed precision if True.
        num_workers: dataloader workers.
        seed: random seed.

    Misc:
        logging_steps: steps between train log prints.
        save_steps: (not used in custom loop; we save each epoch).
    """

    # Model & data
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dataset_name: str = "yahma/alpaca-cleaned"
    dataset_dir: str = "./data"
    train_split: str = "train"
    val_split: str = "validation"
    val_fraction: float = 0.02
    test_split: Optional[str] = "test"
    test_fraction: float = 0.02
    text_field: str = "text"
    max_seq_length: int = 256
    use_legacy_tokenizer: bool = False

    # LoRA
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None

    # Training
    output_dir: str = "./checkpoints"
    epochs: int = 3
    batch_size: int = 4
    grad_accum_steps: int = 16
    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    fp16: bool = True
    num_workers: int = 16
    seed: int = 42

    # Logging
    logging_steps: int = 50
    save_steps: int = 500

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict copy of the config."""
        d = asdict(self)
        return d

    def __post_init__(self) -> None:
        """Fill defaults for target_modules."""
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
