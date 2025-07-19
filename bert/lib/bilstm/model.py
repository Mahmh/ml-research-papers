from typing import Optional, Literal, Tuple
from datetime import datetime
from contextlib import nullcontext
from pathlib import Path
import json, time

from transformers import (
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
    SchedulerType,
)
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch, torch.nn as nn, torch.nn.functional as F

from lib.bilstm.config import BiLSTMConfig
from lib.utils import DEVICE, Metrics, save_checkpoint, plot_loss_curve, make_metadata


class BiLSTMClassifier(nn.Module):
    def __init__(self, config: BiLSTMConfig):
        super().__init__()
        emb_dim = config.embedding_dim
        hid_dim = config.hidden_dim
        layers = config.num_layers

        self.embedding = nn.Embedding(config.vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=layers,
            dropout=config.dropout if layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        lstm_out = hid_dim * 2
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(lstm_out, config.num_labels)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        # embedding + LSTM
        x = self.embedding(input_ids)
        packed_out, (h_n, _c_n) = self.lstm(x)

        # concatenate final forward/backward hidden states
        if self.lstm.bidirectional:
            h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            h = h_n[-1]
        h = self.dropout(h)
        logits = self.fc(h)

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return logits, loss
        return logits


def prepare_optimizer_and_scheduler(
    model: nn.Module, batches_per_epoch: int, config: BiLSTMConfig
) -> Tuple[Optimizer, SchedulerType]:
    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
        eps=config.eps,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=batches_per_epoch * config.num_epochs,
    )

    return optimizer, scheduler


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    desc: str = "Eval",
    training: bool = False,
) -> Metrics:
    model.eval() if not training else model.train()
    ctx = nullcontext() if training else torch.no_grad()

    total_loss, all_preds, all_labels = 0.0, [], []
    with ctx:
        for batch in tqdm(dataloader, desc=desc, leave=False):
            inputs = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            output = model(inputs)
            loss = F.cross_entropy(output, labels)
            total_loss += loss.item()

            preds = torch.argmax(output, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return Metrics(avg_loss, accuracy, precision, recall, f1)


def train(
    model: BiLSTMClassifier,
    optimizer: Optimizer,
    scheduler: SchedulerType,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: BiLSTMConfig,
    model_name: Optional[str] = None,
    step_num: Optional[int] = None,
) -> None:
    model.to(DEVICE)

    # setup run dir
    name = model_name or model.__class__.__name__
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_root = Path(__file__).parents[2] / "checkpoints"
    run_dir = ckpt_root / name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"-> Checkpoints & plots in {run_dir}")

    best_val_f1 = 0.0
    train_losses, val_losses, epoch_times = [], [], []
    step_num = step_num or 0
    t_start = time.perf_counter()

    for epoch in range(1, config.num_epochs + 1):
        e_start = time.perf_counter()

        # training phase
        model.train()
        for batch in tqdm(
            train_loader, desc=f"Train {epoch}/{config.num_epochs}", leave=False
        ):
            inputs = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            _, loss = model(inputs, labels=labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            step_num += 1

        # training & validation metrics
        train_metrics = evaluate(
            model,
            train_loader,
            desc=f"TrainEval {epoch}/{config.num_epochs}",
            training=False,
        )
        val_metrics = evaluate(
            model, val_loader, desc=f"Eval      {epoch}/{config.num_epochs}"
        )
        train_losses.append(train_metrics.loss)
        val_losses.append(val_metrics.loss)
        print(
            f"[Epoch {epoch}/{config.num_epochs}] "
            f"Train Loss: {train_metrics.loss:.4f} | Train F1: {train_metrics.f1:.4f} | "
            f"Val Loss: {val_metrics.loss:.4f} | Val F1: {val_metrics.f1:.4f} "
            f"(best f1: {best_val_f1:.4f})"
        )

        # checkpoints
        elapsed = time.perf_counter() - t_start
        if val_metrics.f1 > best_val_f1:
            best_val_f1 = val_metrics.f1
            path = run_dir / f"model_best_epoch{epoch}.pt"
            save_checkpoint(
                model,
                optimizer,
                str(path),
                epoch=epoch,
                step=step_num,
                metadata=make_metadata(
                    train_metrics, val_metrics, epoch_times, elapsed, timestamp, config
                ),
            )
            print(f"-> New best checkpoint saved to {path}")
        if epoch % config.save_every == 0 or epoch == config.num_epochs:
            path = run_dir / f"model_epoch{epoch}.pt"
            save_checkpoint(
                model,
                optimizer,
                str(path),
                epoch=epoch,
                step=step_num,
                metadata=make_metadata(
                    train_metrics, val_metrics, epoch_times, elapsed, timestamp, config
                ),
            )
            print(f"-> Periodic checkpoint saved to {path}")

        # loss curve
        plot_loss_curve(
            train_losses,
            val_losses,
            "BiLSTM Training Loss Curve",
            str(run_dir / "loss_curve.png"),
        )
        epoch_times.append(time.perf_counter() - e_start)

    # final test
    test_metrics = evaluate(model, test_loader, desc="Test")
    print(
        f"[Test] Loss: {test_metrics.loss:.4f} | "
        f"Acc: {test_metrics.accuracy:.4f} | "
        f"Prec: {test_metrics.precision:.4f} | "
        f"Rec: {test_metrics.recall:.4f} | "
        f"F1: {test_metrics.f1:.4f}"
    )

    # include test metrics in metadata
    final_meta = make_metadata(
        train_metrics,
        val_metrics,
        epoch_times,
        time.perf_counter() - t_start,
        timestamp,
        config,
        test_metrics,
    )
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(final_meta, f, indent=2)
    print(f"-> Final metadata saved to {run_dir}/metadata.json")


def infer(
    model: nn.Module, text: str, config: BiLSTMConfig
) -> Literal["Positive", "Negative"]:
    model.to(DEVICE).eval()
    tokenizer = BertTokenizerFast.from_pretrained(config.tokenizer_name)

    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=config.max_seq_length,
        return_tensors="pt",
    )

    inputs = enc["input_ids"].to(DEVICE)
    with torch.no_grad():
        logits = model(inputs)
    pred = logits.argmax(dim=-1).item()

    return "Positive" if pred == 1 else "Negative"
