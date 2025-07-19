from typing import Optional, Tuple, List, Type
from dataclasses import dataclass
from torch.optim import Optimizer
from torch.utils.data import Dataset
from datasets import load_dataset
import os, json, glob, torch, torch.nn as nn, matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class BaseConfig:
    batch_size: int
    num_epochs: int
    learning_rate: float
    num_labels: int
    save_every: int
    seed: int


@dataclass
class Metrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def load_imdb(
    val_fraction: float = 0.1, seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """Load the stanfordnlp/imdb dataset and split it into train, validation, and test."""
    # load all splits (train, test, unsupervised)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(current_dir, "../data")
    ds = load_dataset("stanfordnlp/imdb", cache_dir=cache_dir)

    # carve off `validation` from the original train
    split = ds["train"].train_test_split(
        test_size=val_fraction,
        seed=seed,
        shuffle=True,
    )

    train_ds = split["train"]
    val_ds = split["test"]
    test_ds = ds["test"]

    return train_ds, val_ds, test_ds


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    path: str,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save model + optimizer states (and optionally epoch & step) to `path`.
    Also, if `metadata` is provided, dump it as metadata.json next to `path`.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if step is not None:
        checkpoint["step"] = step

    torch.save(checkpoint, path)

    if metadata is not None:
        meta_path = os.path.join(os.path.dirname(path), "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)


def load_checkpoint(
    checkpoint_root: str,
    model_class: Type[nn.Module],
    config_class: Type[BaseConfig],
    optimizer: Optional[Optimizer] = None,
) -> Tuple[nn.Module, Optional[Optimizer], Optional[int], Optional[int], BaseConfig]:
    """
    Load a run-directory checkpoint:
      - Reads `metadata.json` in `checkpoint_root` to rebuild the config.
      - Instantiates `model_class(config)` and moves it to `device`.
      - Picks the best checkpoint file (model_best_epoch*.pt), falling back
        to the latest periodic checkpoint (model_epoch*.pt).
      - Loads model & optimizer state_dicts, and returns:
         (model, optimizer_or_None, loaded_epoch, loaded_step, loaded_config)
    """
    # load metadata + config
    meta_path = os.path.join(checkpoint_root, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.json not found in {checkpoint_root}")

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    config_dict = metadata.get("config")

    if config_dict is None:
        raise LookupError("Config not found in metadata.json.")

    config = config_class(**config_dict)

    # instantiate model
    model = model_class(config).to(DEVICE)

    # find checkpoint file
    best_files = glob.glob(os.path.join(checkpoint_root, "model_best_epoch*.pt"))
    if best_files:
        ckpt_path = sorted(best_files)[-1]
    else:
        epoch_files = glob.glob(os.path.join(checkpoint_root, "model_epoch*.pt"))
        if not epoch_files:
            raise FileNotFoundError(f"No .pt files found in {checkpoint_root}")
        ckpt_path = sorted(epoch_files)[-1]

    # load checkpoint
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    loaded_epoch = ckpt.get("epoch")
    loaded_step = ckpt.get("step")

    # load optimizer if provided
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return model, optimizer, loaded_epoch, loaded_step, config

    return model, None, loaded_epoch, loaded_step, config


def plot_loss_curve(train_losses, val_losses, title: str, save_path: str):
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure()
    plt.plot(epochs, train_losses, label="training loss")
    plt.plot(epochs, val_losses, label="validation loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def make_metadata(
    train_metrics: Metrics,
    val_metrics: Metrics,
    epoch_times: List[float],
    total_time: float,
    timestamp: str,
    config: BaseConfig,
    test_metrics: Optional[Metrics] = None,
):
    return {
        "config": vars(config),
        "train_metrics": vars(train_metrics),
        "val_metrics": vars(val_metrics),
        "test_metrics": vars(test_metrics) if test_metrics else None,
        "time_per_epoch": (
            sum(epoch_times) / len(epoch_times) if len(epoch_times) > 0 else None
        ),
        "total_time": total_time,
        "timestamp": timestamp,
    }
