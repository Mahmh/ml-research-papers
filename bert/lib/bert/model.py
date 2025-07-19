from typing import Tuple, Optional, Literal
from datetime import datetime
from contextlib import nullcontext
from pathlib import Path
import os, time, json

from torch import Tensor
from torch.optim import Optimizer, AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch, torch.nn as nn

from transformers import (
    SchedulerType,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
    PreTrainedModel,
    BertTokenizerFast,
)

from lib.bert.config import BERTConfig
from lib.utils import (
    DEVICE,
    Metrics,
    save_checkpoint,
    plot_loss_curve,
    make_metadata,
)


class BERTClassifier(nn.Module):
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            config.pretrained_model_name,
            num_labels=config.num_labels,
            hidden_dropout_prob=config.dropout_prob,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        labels: Optional[Tensor] = None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )


def prepare_optimizer_and_scheduler(
    model: BERTClassifier, batches_per_epoch: int, config: BERTConfig
) -> Tuple[Optimizer, SchedulerType]:
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_params = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_params,
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=batches_per_epoch * config.num_epochs,
    )

    return optimizer, scheduler


def train(
    model: BERTClassifier,
    optimizer: Optimizer,
    scheduler: SchedulerType,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: BERTConfig,
    model_name: Optional[str] = None,
    step_num: Optional[int] = None,
) -> None:
    model.to(DEVICE)

    # setup run dir
    model_name = model_name or model.__class__.__name__
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_root = os.path.join(current_dir, "../../checkpoints")
    run_dir = Path(checkpoint_root) / model_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"-> Checkpoints, plots & metadata in {run_dir}")

    # containers for losses & timing
    best_val_f1 = 0.0
    train_losses, val_losses, epoch_times = [], [], []
    step_num = step_num or 0
    t_start = time.perf_counter()

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.perf_counter()

        # run the training pass
        model.train()
        for batch in tqdm(
            train_loader, desc=f"Train {epoch}/{config.num_epochs}", leave=False
        ):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = outputs.loss

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
            training=True,
        )
        val_metrics = evaluate(
            model, val_loader, desc=f"Eval  {epoch}/{config.num_epochs}"
        )
        train_losses.append(train_metrics.loss)
        val_losses.append(val_metrics.loss)
        print(
            f"[Epoch {epoch}/{config.num_epochs}] "
            f"Train Loss: {train_metrics.loss:.4f} | Train F1: {train_metrics.f1:.4f} | "
            f"Val Loss: {val_metrics.loss:.4f} | Val F1: {val_metrics.f1:.4f} "
            f"(best f1: {best_val_f1:.4f})"
        )

        # best‐val checkpoint
        if val_metrics.f1 > best_val_f1:
            best_val_f1 = val_metrics.f1
            best_ckpt = run_dir / f"model_best_epoch{epoch}.pt"
            total_time = time.perf_counter() - t_start
            save_checkpoint(
                model,
                optimizer,
                str(best_ckpt),
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
            print(f"-> New best checkpoint saved to {best_ckpt}")

        # periodic checkpoint
        if epoch % config.save_every == 0 or epoch == config.num_epochs:
            ckpt = run_dir / f"model_epoch{epoch}.pt"
            total_time = time.perf_counter() - t_start
            save_checkpoint(
                model,
                optimizer,
                str(ckpt),
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
            print(f"-> Periodic checkpoint saved to {ckpt}")

        loss_plot = run_dir / "loss_curve.png"
        plot_loss_curve(
            train_losses, val_losses, "BERT Training Loss Curve", str(loss_plot)
        )
        epoch_times.append(time.perf_counter() - epoch_start)

    # test
    test_metrics = evaluate(model, test_loader, desc="Test")
    print(
        f"[Test] Loss: {test_metrics.loss:.4f} | "
        f"Acc: {test_metrics.accuracy:.4f} | "
        f"Prec: {test_metrics.precision:.4f} | "
        f"Rec: {test_metrics.recall:.4f} | "
        f"F1: {test_metrics.f1:.4f}"
    )

    # overwrite metadata.json to include test_metrics
    total_time = time.perf_counter() - t_start
    final_meta = make_metadata(
        train_metrics,
        val_metrics,
        epoch_times,
        total_time,
        timestamp,
        config,
        test_metrics,
    )
    meta_path = run_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(final_meta, f, indent=2)
    print(f"-> Final metadata saved to {meta_path}")


def evaluate(
    model: BERTClassifier,
    dataloader: DataLoader,
    desc: str = "Eval",
    training: bool = False,
) -> Metrics:
    model.eval() if not training else model.train()
    ctx = nullcontext() if training else torch.no_grad()

    total_loss, all_preds, all_labels = 0.0, [], []
    with ctx:
        for batch in tqdm(dataloader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = outputs.loss
            logits = outputs.logits

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


def infer(
    model: PreTrainedModel, text: str, config: BERTConfig
) -> Literal["Positive", "Negative"]:
    # set up DEVICE & model
    model.to(DEVICE).eval()

    # load tokenizer matching the model
    tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_model_name)

    # preprocess
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=config.max_seq_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # inference
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = int(torch.argmax(outputs.logits, dim=-1).item())

    return "Positive" if pred_id == 1 else "Negative"
