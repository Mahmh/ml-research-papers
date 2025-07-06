from typing import List, Callable, Any, Tuple, Iterator
from datasets import load_dataset
from transformers import AutoTokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.transforms import Sequential, AddToken, ToTensor, Truncate, VocabTransform
from torch.utils.data import DataLoader
from torch import Tensor
import torch.nn as nn
from lib.constants import Config, SPECIAL_TOKENS

# 1. Load tokenizer & dataset
ds = load_dataset('bentrevett/multi30k', cache_dir='./multi30k')
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")


# 2. Preprocessing
def _yield_tokens(split: str, lang: str) -> Iterator[List[str]]:
    """
    Iterate through a Hugging Face Dataset split and yield token lists for each translation example.

    Args:
        split (str): Which dataset split to process (e.g., 'train', 'validation', 'test').
        lang (str): Language code of the text to tokenize (e.g., 'en', 'de').

    Yields:
        List[str]: The sequence of tokens produced by `tokenizer.tokenize()` for each example
                   in the specified split and language.
    """
    for example in ds[split]:
        yield tokenizer.tokenize(example[lang])


vocab_src = build_vocab_from_iterator(_yield_tokens('train', 'en'), specials=SPECIAL_TOKENS)
vocab_tgt = build_vocab_from_iterator(_yield_tokens('train', 'de'), specials=SPECIAL_TOKENS)
vocab_src.set_default_index(vocab_src['<unk>'])
vocab_tgt.set_default_index(vocab_tgt['<unk>'])


class TokenizerTransform(nn.Module):
    """
    PyTorch transform module that applies a tokenization function to each element in a batch.

    ## Args:
        tok (Callable[[Any], Any]): A tokenization callable (e.g., `tokenizer.encode` or `tokenizer.tokenize`).
    """
    def __init__(self, tok: Callable[[Any], Any]) -> None:
        super().__init__()
        self.tok = tok

    def forward(self, batch: List[Any]) -> List[Any]:
        """
        Apply the tokenization function over a batch of inputs.

        ## Args:
            batch (List[Any]): Iterable of input items (e.g., strings) to be tokenized.

        ## Returns:
            List[Any]: A list where each element is the tokenized output corresponding to the input batch element.
        """
        return [self.tok(x) for x in batch]


def _make_pipeline(tok: Callable[[str], Any], vocab: dict, max_len: int) -> Sequential:
    """
    Creates a processing pipeline that:
      - tokenizes raw strings
      - truncates to (max_len - 2) tokens
      - adds <bos> and <eos> tokens
      - maps tokens to integer IDs via vocab
      - converts to padded tensor

    ## Args:
        tok (Callable[[str], Any]): Tokenizer function mapping strings to token sequences.
        vocab (dict): Vocabulary mapping tokens (str) to integer IDs.
        max_len (int): Maximum output sequence length, including BOS/EOS tokens.

    ## Returns:
        Sequential: A composed transform applying tokenization, truncation, token addition,
                    vocabulary mapping, and tensor conversion.
    """
    return Sequential(
        TokenizerTransform(tok),
        Truncate(max_len - 2),
        AddToken('<bos>', begin=True),
        AddToken('<eos>', begin=False),
        VocabTransform(vocab),
        ToTensor(padding_value=vocab['<pad>'])
    )


def _collate_fn(src_pipeline: Sequential, tgt_pipeline: Sequential) -> Callable[[List[dict]], Tuple[Tensor, Tensor, Tensor]]:
    """
    Builds a collate function for DataLoader that applies source/target pipelines and splits target into input and labels.

    ## Args:
        src_pipeline (Sequential): Pipeline to process source text strings.
        tgt_pipeline (Sequential): Pipeline to process target text strings.

    ## Returns:
        Callable: A function that takes a batch of examples and returns a tuple: (src_tensor, tgt_input, tgt_labels).
    """
    def collate(batch: List[dict]) -> Tuple[Tensor, Tensor, Tensor]:
        src_raw = [row['en'] for row in batch]  # English source
        tgt_raw = [row['de'] for row in batch]  # German target
        src_tensor = src_pipeline(src_raw)
        tgt_tensor = tgt_pipeline(tgt_raw)
        tgt_input  = tgt_tensor[:, :-1]
        tgt_labels = tgt_tensor[:, 1:]
        return src_tensor, tgt_input, tgt_labels
    return collate


def get_data_loaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs DataLoaders for the training, validation, and test splits using configured pipelines.

    ## Args:
        config (Config): Configuration instance containing batch size and sequence length settings.

    ## Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: DataLoaders for train, validation, and test sets.
    """
    src_pipe = _make_pipeline(tokenizer.tokenize, vocab_src, config.input_seq_len)
    tgt_pipe = _make_pipeline(tokenizer.tokenize, vocab_tgt, config.output_seq_len)
    collate  = _collate_fn(src_pipe, tgt_pipe)

    train_loader = DataLoader(ds["train"], batch_size=config.batch_size, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(ds["validation"], batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(ds["test"], batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader, test_loader