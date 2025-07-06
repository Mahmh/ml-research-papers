from typing import Iterator, List, Callable, Any, Tuple
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.transforms import Sequential, AddToken, ToTensor, Truncate, VocabTransform
from torch.utils.data import DataLoader
from torch import Tensor
import torch.nn as nn
from lib.constants import Config, SPECIAL_TOKENS, MAX_NUM_EXAMPLES, DATA_DIR, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT

# 1. Load tokenizer & dataset
tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-cs-en', use_fast=True)
full_ds = load_dataset('wmt/wmt19', 'cs-en', split='train', cache_dir=DATA_DIR)
ds = full_ds.select(range(MAX_NUM_EXAMPLES)) if MAX_NUM_EXAMPLES < len(full_ds) else full_ds
ds = ds.shuffle(seed=42)


# 2. Split dataset into training, validation, and testing splits
holdout_frac = 1.0 - TRAIN_SPLIT
split1 = ds.train_test_split(test_size=holdout_frac, seed=42)
train_ds = split1['train']
temp_ds = split1['test']

val_frac_of_temp = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
split2 = temp_ds.train_test_split(test_size=val_frac_of_temp, seed=42)
val_ds = split2['test']
test_ds = split2['train']


# 3. Preprocessing
def _yield_tokens(dataset: Dataset, lang: str) -> Iterator[List[str]]:
    """
    Iterate through a HF Dataset and yield token lists for each translation example.

    ## Args:
        dataset (datasets.Dataset): A Hugging Face Dataset containing translation examples,
            where each example has a `translation` field mapping language codes to text.
        lang (str): Language code to extract and tokenize ('en' for English or 'cs' for Czech).

    ## Yields:
        List[str]: The sequence of tokens produced by `tokenizer.tokenize` for the
        specified language text of each example.
    """
    for row in dataset:
        yield tokenizer.tokenize(row['translation'][lang])


vocab_src = build_vocab_from_iterator(_yield_tokens(train_ds, 'en'), specials=SPECIAL_TOKENS)
vocab_tgt = build_vocab_from_iterator(_yield_tokens(train_ds, 'cs'), specials=SPECIAL_TOKENS)
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
        src_raw = [row['translation']['en'] for row in batch]  # English source
        tgt_raw = [row['translation']['cs'] for row in batch]  # Czech target
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

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader, test_loader