from datetime import datetime
from typing import Optional, Literal
from contextlib import nullcontext
import os, json, time

from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from lib.utils import DEVICE, Metrics, save_checkpoint, plot_loss_curve, make_metadata
from lib.lr.config import LogisticRegressionConfig
import torch, torch.nn as nn, torch.nn.functional as F


class LogisticRegressionClassifier(nn.Module):
    def __init__(self, config: LogisticRegressionConfig):
        super().__init__()
        self.linear = nn.Linear(config.input_dim, config.num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def prepare_optimizer(model: nn.Module, config: LogisticRegressionConfig) -> Optimizer:
    return SGD(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def train(
    model: LogisticRegressionClassifier,
    optimizer: Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: LogisticRegressionConfig,
    model_name: Optional[str] = None,
    step_num: Optional[int] = None,
) -> None:
    model.to(DEVICE)

    # setup run dir
    model_name = model_name or model.__class__.__name__
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_root = os.path.join(os.path.dirname(__file__), "../../checkpoints")
    run_dir = os.path.join(ckpt_root, model_name, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"-> Checkpoints & plots in {run_dir}")

    best_val_f1 = 0.0
    train_losses, val_losses, epoch_times = [], [], []
    step_num = step_num or 0
    t_start = time.perf_counter()

    for epoch in range(1, config.num_epochs + 1):
        ep_start = time.perf_counter()
        # training
        model.train()
        for x, labels in tqdm(
            train_loader, desc=f"Train {epoch}/{config.num_epochs}", leave=False
        ):
            x, labels = x.to(DEVICE), labels.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_num += 1

        # metrics
        train_metrics = evaluate(
            model,
            train_loader,
            desc=f"TrainEval {epoch}/{config.num_epochs}",
            training=True,
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
        total_time = time.perf_counter() - t_start
        if val_metrics.f1 > best_val_f1:
            best_val_f1 = val_metrics.f1
            path = os.path.join(run_dir, f"model_best_epoch{epoch}.pt")
            save_checkpoint(
                model,
                optimizer,
                path,
                epoch=epoch,
                step=step_num,
                metadata=make_metadata(
                    train_metrics,
                    val_metrics,
                    epoch_times,
                    total_time,
                    timestamp,
                    config,
                ),
            )
            print(f"-> New best checkpoint saved to {path}")

        if epoch % config.save_every == 0 or epoch == config.num_epochs:
            path = os.path.join(run_dir, f"model_epoch{epoch}.pt")
            save_checkpoint(
                model,
                optimizer,
                path,
                epoch=epoch,
                step=step_num,
                metadata=make_metadata(
                    train_metrics,
                    val_metrics,
                    epoch_times,
                    total_time,
                    timestamp,
                    config,
                ),
            )
            print(f"-> Periodic checkpoint saved to {path}")

        # loss curve
        plot_loss_curve(
            train_losses,
            val_losses,
            "Logistic Regression Training Loss Curve",
            os.path.join(run_dir, "loss_curve.png"),
        )
        epoch_times.append(time.perf_counter() - ep_start)

    # final test
    test_metrics = evaluate(model, test_loader, desc="Test")
    print(
        f"[Test] Loss: {test_metrics.loss:.4f} | "
        f"Acc: {test_metrics.accuracy:.4f} | "
        f"Prec: {test_metrics.precision:.4f} | "
        f"Rec: {test_metrics.recall:.4f} | "
        f"F1: {test_metrics.f1:.4f}"
    )

    # save final metadata
    final_meta = make_metadata(
        train_metrics,
        val_metrics,
        epoch_times,
        time.perf_counter() - t_start,
        timestamp,
        config,
        test_metrics,
    )
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(final_meta, f, indent=2)
    print(f"-> Final metadata saved to {run_dir}/metadata.json")


def evaluate(
    model: nn.Module, dataloader: DataLoader, desc: str = "Eval", training: bool = False
) -> Metrics:
    model.eval() if not training else model.train()
    ctx = nullcontext() if training else torch.no_grad()

    total_loss = 0.0
    all_preds, all_labels = [], []
    with ctx:
        for x, labels in tqdm(dataloader, desc=desc, leave=False):
            x = x.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(x)
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return Metrics(avg_loss, accuracy, precision, recall, f1)


def infer(model: nn.Module, vectorizer, text: str) -> Literal["Positive", "Negative"]:
    model.to(DEVICE).eval()
    x = vectorizer.transform([text])
    x = torch.from_numpy(x.toarray()).float().to(DEVICE)

    with torch.no_grad():
        logits = model(x)
    pred = torch.argmax(logits, dim=-1).item()

    return "Positive" if pred == 1 else "Negative"
