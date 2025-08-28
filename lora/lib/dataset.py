from typing import Tuple, Optional, Dict
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    DataCollatorForLanguageModeling,
)
from lib.config import Config


def _get_tokenizer(cfg: Config) -> PreTrainedTokenizerBase:
    """
    Load tokenizer and ensure pad token is set (important for CausalLM training).
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, use_fast=True, use_legacy=cfg.use_legacy_tokenizer
        )
    except ValueError:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, use_fast=False, use_legacy=cfg.use_legacy_tokenizer
        )
    if tokenizer.pad_token is None:
        # Reuse EOS as PAD if absent (common for decoder-only models)
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _prepare_split(
    ds: Dataset, cfg: Config, tokenizer: PreTrainedTokenizerBase
) -> Dataset:
    """
    Tokenize and set torch format for a single split.
    Adds 'labels' identical to 'input_ids' (Model will shift internally).
    """

    def tokenize_fn(batch: Dict) -> Dict:
        instrs = batch.get("instruction", [])
        inps = batch.get("input", [])
        outs = batch.get("output", [])

        # build prompts in bulk
        prompts = [process_prompt(i, x) for i, x in zip(instrs, inps)]

        # tokenize prompts and outputs in bulk (FAST tokenizer path)
        enc_p = tokenizer(prompts, add_special_tokens=False)
        enc_y = tokenizer(outs, add_special_tokens=False)

        max_len = cfg.max_seq_length
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        eos_id = tokenizer.eos_token_id

        input_ids_list, attn_list, labels_list = [], [], []

        for p_ids, y_ids in zip(enc_p["input_ids"], enc_y["input_ids"]):
            # ensure target ends with EOS
            if eos_id is not None and (len(y_ids) == 0 or y_ids[-1] != eos_id):
                y_ids = y_ids + [eos_id]

            input_ids = p_ids + y_ids
            labels = [-100] * len(p_ids) + y_ids
            attn = [1] * len(input_ids)

            # truncate
            if len(input_ids) > max_len:
                cut = len(input_ids) - max_len
                if cut < len(y_ids):
                    # trim from start of target
                    y_ids = y_ids[cut:]
                    input_ids = p_ids + y_ids
                    labels = [-100] * len(p_ids) + y_ids
                    attn = [1] * len(input_ids)
                else:
                    # prompt itself too long -> hard cut
                    input_ids = input_ids[-max_len:]
                    labels = labels[-max_len:]
                    attn = attn[-max_len:]

            # pad
            pad = max_len - len(input_ids)
            if pad > 0:
                input_ids += [pad_id] * pad
                attn += [0] * pad
                labels += [-100] * pad

            input_ids_list.append(input_ids)
            attn_list.append(attn)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attn_list,
            "labels": labels_list,
        }

    # Use batched mapping (+ parallel workers for speed)
    ds = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=cfg.num_workers,
    )

    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def process_prompt(instruction: str, input: Optional[str] = None):
    """Makes the input prompt consistent between training and inference."""
    return (
        f"Input: {input}\nInstruction: {instruction}"
        if input
        else f"Instruction: {instruction}"
    )


def get_datasets(
    cfg: Config,
) -> Tuple[
    Dataset,
    Dataset,
    Optional[Dataset],
    PreTrainedTokenizerBase,
    DataCollatorForLanguageModeling,
]:
    """Load dataset, ensure val/test exist by carving both from train if needed, tokenize, return collator."""
    tokenizer = _get_tokenizer(cfg)
    dset = load_dataset(cfg.dataset_name, cache_dir=cfg.dataset_dir)

    if cfg.train_split not in dset:
        raise ValueError(f"Train split '{cfg.train_split}' not found.")
    train_raw = dset[cfg.train_split]

    val_raw = dset[cfg.val_split] if cfg.val_split in dset else None
    test_raw = dset[cfg.test_split] if cfg.test_split in dset else None

    # If either is missing, carve BOTH from the original train (keeps it simple & disjoint)
    if val_raw is None or test_raw is None:
        val_frac = cfg.val_fraction
        test_frac = cfg.test_fraction
        holdout = train_raw.train_test_split(
            test_size=val_frac + test_frac, seed=cfg.seed
        )
        train_raw = holdout["train"]

        # split holdout into val/test by their ratio
        vt = holdout["test"].train_test_split(
            test_size=test_frac / (val_frac + test_frac), seed=cfg.seed
        )
        val_raw, test_raw = vt["train"], vt["test"]

    train_ds = _prepare_split(train_raw, cfg, tokenizer)
    val_ds = _prepare_split(val_raw, cfg, tokenizer)
    test_ds = _prepare_split(test_raw, cfg, tokenizer) if test_raw is not None else None

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return train_ds, val_ds, test_ds, tokenizer, collator
