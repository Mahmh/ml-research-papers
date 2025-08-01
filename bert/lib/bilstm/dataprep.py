from typing import Tuple, Dict, List
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from lib.utils import load_imdb
from lib.bilstm.config import BiLSTMConfig
import torch


class IMDbDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: BertTokenizerFast,
        max_seq_length: int,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        # squeeze batch dimension
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def get_data_loaders(config: BiLSTMConfig) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    # load raw splits
    train_ds, val_ds, test_ds = load_imdb()

    # instantiate tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(config.tokenizer_name)

    # build dataset objects
    train_dataset = IMDbDataset(
        texts=train_ds["text"],
        labels=train_ds["label"],
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
    )
    val_dataset = IMDbDataset(
        texts=val_ds["text"],
        labels=val_ds["label"],
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
    )
    test_dataset = IMDbDataset(
        texts=test_ds["text"],
        labels=test_ds["label"],
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
    )

    # wrap in DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, tokenizer.vocab_size
