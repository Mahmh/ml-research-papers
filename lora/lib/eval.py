from __future__ import annotations
from typing import Dict
from transformers import PreTrainedModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from evaluate import load as load_metric
import math, time, torch

# Load text-based metrics
bleu_metric = load_metric("sacrebleu")
rouge_metric = load_metric("rouge")
meteor_metric = load_metric("meteor")


def evaluate_lm(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device,
    tokenizer,
    max_gen_len: int = 128,
    use_amp: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a CausalLM with both LM metrics and text-based metrics if references exist.

    LM metrics:
        loss, perplexity, token_accuracy, bits_per_token, tokens_per_second

    Text generation metrics (if labels exist as text):
        BLEU, ROUGE-L, METEOR
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    start = time.time()

    # For text metrics
    all_preds_text = []
    all_refs_text = []

    autocast_dtype = torch.float16 if use_amp and device.type == "cuda" else None
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
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
            else:
                out = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

            loss = out.loss
            logits = out.logits

            valid = labels != -100
            total_loss += loss.item() * valid.sum().item()
            total_tokens += valid.sum().item()

            preds = logits.argmax(dim=-1)
            correct_tokens += (preds[valid] == labels[valid]).sum().item()

            ignore_index = -100
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = (
                    tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                )

            labels_for_decode = labels.detach().clone()
            labels_for_decode[labels_for_decode == ignore_index] = pad_id

            # Text metrics: generate predictions & decode
            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen_len,
                do_sample=False,
            )
            new_tokens = gen_out[:, input_ids.size(1) :]  # slice off the prompt
            pred_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            ref_texts = tokenizer.batch_decode(
                labels_for_decode, skip_special_tokens=True
            )

            all_preds_text.extend(pred_texts)
            all_refs_text.extend(ref_texts)

    elapsed = max(time.time() - start, 1e-9)
    mean_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    token_acc = correct_tokens / max(total_tokens, 1)
    bpt = mean_loss / math.log(2.0)
    tps = total_tokens / elapsed

    # Compute BLEU, ROUGE-L, METEOR
    bleu = bleu_metric.compute(
        predictions=all_preds_text, references=[[r] for r in all_refs_text]
    )["score"]
    rouge = rouge_metric.compute(predictions=all_preds_text, references=all_refs_text)
    meteor = meteor_metric.compute(
        predictions=all_preds_text, references=all_refs_text
    )["meteor"]

    rouge_l = rouge["rougeL"]

    return {
        "loss": float(mean_loss),
        "perplexity": float(ppl),
        "token_accuracy": float(token_acc),
        "bits_per_token": float(bpt),
        "tokens_per_second": float(tps),
        "BLEU": float(bleu),
        "ROUGE-L": float(rouge_l),
        "METEOR": float(meteor),
    }
