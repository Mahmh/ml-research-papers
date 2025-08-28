from __future__ import annotations
from typing import Any, Dict, Mapping, Optional, Union, Tuple
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import asdict, is_dataclass

from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lib.config import Config
from lib.dataset import get_datasets
from lib.model import get_model, get_device, count_params
from lib.eval import evaluate_lm

import json, time, math, torch, matplotlib, matplotlib.pyplot as plt

matplotlib.use("Agg")


def _cfg_to_dict(cfg: Config) -> Dict[str, Any]:
    if hasattr(cfg, "as_dict"):
        return cfg.as_dict()
    if is_dataclass(cfg):
        return asdict(cfg)
    try:
        return dict(cfg)
    except Exception:
        raise TypeError(
            "Config must implement .as_dict(), be a dataclass, or be Mapping-like."
        )


def timestamp_str() -> str:
    """Return UTC timestamp suitable for folder names (with microseconds to avoid collisions)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")


def get_metadata(
    ts: str,
    epoch: int,
    global_step: int,
    cfg: Config,
    total_params: int,
    trainable_params: int,
    train_epoch_loss: float,
    val_metrics: Optional[Mapping[str, Any]],
    test_metrics: Optional[Mapping[str, Any]],
    epoch_time: float,
    total_time: float,
    device: Union[torch.device, str],
    peak_vram_mib: Optional[float],
    ckpt_dir: Path,
) -> Dict[str, Any]:
    """Create a training metadata dict (JSON-serializable). All parameters are positional."""
    cfg_dict = _cfg_to_dict(cfg)

    return {
        "timestamp": ts,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": cfg_dict,
        "params": {
            "total": int(total_params),
            "trainable": int(trainable_params),
            "trainable_pct": float(100.0 * trainable_params / max(1, total_params)),
        },
        "train": {
            "loss": float(train_epoch_loss),
        },
        "validation": (
            {
                "loss": float(val_metrics["loss"]),
                "perplexity": float(val_metrics["perplexity"]),
                "token_accuracy": float(val_metrics["token_accuracy"]),
                "bits_per_token": float(val_metrics["bits_per_token"]),
                "tokens_per_second": float(val_metrics["tokens_per_second"]),
            }
            if val_metrics is not None
            else None
        ),
        "test": (
            {
                "loss": float(test_metrics["loss"]),
                "perplexity": float(test_metrics["perplexity"]),
                "token_accuracy": float(test_metrics["token_accuracy"]),
                "bits_per_token": float(test_metrics["bits_per_token"]),
                "tokens_per_second": float(test_metrics["tokens_per_second"]),
            }
            if test_metrics is not None
            else None
        ),
        "times": {
            "epoch_seconds": float(epoch_time),
            "total_seconds": float(total_time),
        },
        "resources": {
            "device": str(device),
            "peak_vram_mib": (
                float(peak_vram_mib) if peak_vram_mib is not None else None
            ),
        },
        "artifacts": {
            "loss_curve_png": (ckpt_dir / "loss_curve.png").as_posix(),
        },
    }


def save_checkpoint(
    folder: Path,
    epoch: int,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.amp.GradScaler],
    meta: Dict[str, Any],
) -> Path:
    """
    Save model/optimizer/scheduler/scaler states and metadata into `folder`,
    using a fixed filename `model.pt`. Also writes `metadata.json`.
    """
    folder.mkdir(parents=True, exist_ok=True)
    ckpt_path = folder / "model.pt"
    payload = {
        "epoch": int(epoch),
        "step": int(step),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": (scheduler.state_dict() if scheduler is not None else None),
        "scaler_state": (scaler.state_dict() if scaler is not None else None),
        "meta": meta,
        "torch_version": torch.__version__,
    }
    torch.save(payload, ckpt_path)

    with open(folder / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return ckpt_path


def load_checkpoint(
    ckpt_file: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.amp.GradScaler],
) -> Tuple[int, int, Dict[str, Any]]:
    """
    Load a checkpoint and restore model/optimizer/scheduler/scaler states.
    Returns: (epoch, step, meta_dict)
    """
    payload = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])

    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])

    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])

    epoch = int(payload.get("epoch", 0))
    step = int(payload.get("step", 0))
    meta = payload.get("meta", {})
    return epoch, step, meta


def plot_losses(
    out_path: Path,
    train_losses: list[float],
    val_losses: list[float],
    title: str = "Training & Validation Loss",
) -> None:
    """Save a PNG loss curve to out_path."""
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def train_and_evaluate(cfg: Config, resume_ckpt: Optional[str] = None) -> None:
    """
    Train a CausalLM on a tokenized dataset with optional LoRA adapters.
    Logs metrics, plots loss curves each epoch, and saves **every time** to:
        ./checkpoints/{timestamp}/model.pt
        ./checkpoints/{timestamp}/metadata.json
        ./checkpoints/{timestamp}/loss_curve.png
    """
    torch.manual_seed(cfg.seed)
    device = get_device()

    # Data
    train_ds, val_ds, test_ds, tokenizer, collator = get_datasets(cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collator,
        )

    # Model & optim
    model = get_model(cfg, device)
    total_params, trainable_params = count_params(model)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Scheduler, grad accumulation, & scaler
    micro_steps_per_epoch = len(train_loader)
    opt_steps_per_epoch = math.ceil(
        micro_steps_per_epoch / max(1, cfg.grad_accum_steps)
    )
    total_steps = opt_steps_per_epoch * cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16 and device.type == "cuda")

    # Root output dir (each save uses a fresh timestamped subfolder)
    output_root = Path(cfg.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Trackers
    history_train_loss: list[float] = []
    history_val_loss: list[float] = []
    global_step = 0
    start_time = time.perf_counter()
    start_epoch = 1

    # Defaults so mid-epoch saves have something valid to serialize
    train_epoch_loss: float = 0.0
    val_metrics: Optional[Dict[str, Any]] = None
    test_metrics: Optional[Mapping[str, Any]] = None
    epoch_time: float = 0.0
    total_time: float = 0.0
    peak_mem_mb: Optional[float] = None

    # Optionally resume
    if resume_ckpt is not None:
        ep, global_step, meta = load_checkpoint(
            Path(resume_ckpt), model, optimizer, scheduler, scaler
        )
        start_epoch = ep + 1  # continue with next epoch

    # Training epochs
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        # Token-weighted accumulation
        running_loss_sum = 0.0  # sum over (loss * tokens_in_batch)
        tokens_seen = 0  # total valid tokens
        t0 = time.perf_counter()

        # Reset CUDA peak stats to report VRAM this epoch
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        autocast_dtype = torch.float16 if (cfg.fp16 and device.type == "cuda") else None

        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=True), start=1
        ):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            if autocast_dtype is not None:
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = out.loss
            else:
                out = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = out.loss

            # Convert HF mean loss to token-sum for correct epoch averaging
            loss_value = loss.item()
            tokens_in_batch = (labels != -100).sum().item()
            running_loss_sum += loss_value * tokens_in_batch
            tokens_seen += tokens_in_batch

            # Backward + optimize with grad accumulation
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % cfg.grad_accum_steps == 0:
                if cfg.max_grad_norm is not None:
                    if scaler is not None and scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.max_grad_norm
                    )

                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.logging_steps == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"[epoch {epoch} step {global_step}] loss={loss_value:.4f} lr={lr:.2e}"
                    )

                # Mid-epoch checkpoint saving
                if global_step > 0 and (global_step % cfg.save_steps == 0):
                    train_epoch_loss = running_loss_sum / max(1, tokens_seen)
                    epoch_time = time.perf_counter() - t0
                    total_time = time.perf_counter() - start_time
                    peak_mem_mb = (
                        torch.cuda.max_memory_allocated(device) / (1024**2)
                        if device.type == "cuda"
                        else None
                    )

                    ts_now = timestamp_str()
                    save_dir = output_root / ts_now
                    save_dir.mkdir(parents=True, exist_ok=True)

                    # Plot current curves into this save dir (partial curves mid-epoch)
                    plot_losses(
                        out_path=save_dir / "loss_curve.png",
                        train_losses=history_train_loss,
                        val_losses=history_val_loss,
                        title="Training & Validation Loss",
                    )

                    meta = get_metadata(
                        ts_now,
                        epoch,
                        global_step,
                        cfg,
                        total_params,
                        trainable_params,
                        train_epoch_loss,
                        val_metrics,
                        test_metrics,
                        epoch_time,
                        total_time,
                        device,
                        peak_mem_mb,
                        save_dir,
                    )

                    ckpt_file = save_checkpoint(
                        folder=save_dir,
                        epoch=epoch,
                        step=global_step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        meta=meta,
                    )
                    print(f"Saved checkpoint: {ckpt_file.as_posix()}")
                    print(
                        f"Saved metadata.json at: {(save_dir / 'metadata.json').as_posix()}"
                    )

        # End-of-epoch stats
        train_epoch_loss = running_loss_sum / max(1, tokens_seen)
        history_train_loss.append(train_epoch_loss)

        # Validation
        val_metrics = evaluate_lm(
            model, val_loader, device, tokenizer, use_amp=cfg.fp16
        )
        history_val_loss.append(val_metrics["loss"])

        # Test
        test_metrics = (
            evaluate_lm(model, test_loader, device, tokenizer, use_amp=cfg.fp16)
            if test_loader is not None
            else None
        )

        # Epoch timing and VRAM
        epoch_time = time.perf_counter() - t0
        total_time = time.perf_counter() - start_time
        peak_mem_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else None
        )

        # End-of-epoch checkpoint saving
        ts_now = timestamp_str()
        save_dir = output_root / ts_now
        save_dir.mkdir(parents=True, exist_ok=True)

        # Plot full curves (up to current epoch)
        plot_losses(
            out_path=save_dir / "loss_curve.png",
            train_losses=history_train_loss,
            val_losses=history_val_loss,
            title="Training & Validation Loss",
        )

        meta = get_metadata(
            ts_now,
            epoch,
            global_step,
            cfg,
            total_params,
            trainable_params,
            train_epoch_loss,
            val_metrics,
            test_metrics,
            epoch_time,
            total_time,
            device,
            peak_mem_mb,
            save_dir,
        )

        ckpt_file = save_checkpoint(
            folder=save_dir,
            epoch=epoch,
            step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            meta=meta,
        )
        print(f"Saved checkpoint: {ckpt_file.as_posix()}")
        print(f"Updated metadata.json at: {(save_dir / 'metadata.json').as_posix()}")

    print("Training complete.")
