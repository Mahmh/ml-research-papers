from typing import Tuple
from transformers import AutoModelForCausalLM, PreTrainedModel
from peft import LoraConfig, get_peft_model
from lib.config import Config
import torch


def get_device() -> torch.device:
    """Return CUDA device if available else CPU."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_model(cfg: Config, device: torch.device) -> PreTrainedModel:
    """
    Load a pretrained CausalLM model, optionally with LoRA adapters attached,
    and move it to the specified device.
    """
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    if cfg.use_lora:
        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    model.to(device)
    return model


def count_params(model: PreTrainedModel) -> Tuple[int, int]:
    """
    Count (total, trainable) parameters.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
