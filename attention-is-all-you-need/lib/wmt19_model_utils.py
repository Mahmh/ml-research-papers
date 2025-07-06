from typing import List, Optional, Dict, Tuple
from datetime import datetime
from dataclasses import asdict
from tqdm.auto import tqdm
from torch import device
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os, time, json, torch, torch.optim as optim, matplotlib.pyplot as plt, evaluate
from lib.constants import Config, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
from lib.wmt19_data_utils import get_data_loaders, vocab_src, vocab_tgt, tokenizer
from lib.transformer import Transformer

def _calculate_lr_scale(step_num: int, d_model: int, warmup_steps: float) -> float:
    """
    Computes the Transformer's learning rate scaling factor from 
    "Attention Is All You Need."

    The schedule increases linearly for the first `warmup_steps` steps
    and then decays proportionally to the inverse square root of the step number.

    ## Args:
        step_num (int): Current optimizer step count (should start at 1).
        d_model (int): The hidden size of the Transformer model.
        warmup_steps (float): Number of steps over which to linearly warm up.

    ## Returns:
        float: A scalar multiplier to apply to the base learning rate.
    """
    return (d_model ** -0.5) * min(step_num ** -0.5, step_num * (warmup_steps ** -1.5))


def _save(
    config: Config,
    model: Transformer,
    opt: Optimizer,
    step_num: int,
    ckpt_root: str,
    filename: str,
    epoch_times: List[float],
    train_losses: List[float],
    val_losses: List[float],
    eval_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    Saves the model checkpoint, training history, and optional evaluation metrics.

    ## Args:
        config (Config): Configuration object containing hyperparameters and settings.
        model (Transformer): The Transformer model instance to serialize.
        ckpt_root (str): Path to the checkpoint directory.
        filename (str): Name of the file to which the model’s state_dict will be saved (e.g., "model_epoch3.pt").
        epoch_times (List[float]): List of elapsed times (in seconds) for each completed epoch.
        train_losses (List[float]): Recorded training loss values, one per epoch.
        val_losses (List[float]): Recorded validation loss values, one per epoch.
        eval_metrics (Optional[Dict[str, float]]): Optional dictionary of evaluation metrics
            (e.g., BLEU, ChrF) computed on a hold-out set; these will be included in the metadata.

    ## Side effects:
        - Writes the model's state_dict to "{ckpt_root}/{filename}".
        - Writes a metadata JSON file to "{ckpt_root}/metadata.json" containing:
            * config parameters
            * latest train/val loss
            * list of per-epoch losses
            * epoch timing information
            * any provided evaluation metrics
        - Generates and saves a loss curve plot ("loss_curve.png") in ckpt_root.

    ## Returns:
        None
    """
    torch.save({
        'model_state': model.state_dict(),
        'optim_state': opt.state_dict(),
        'step_num': step_num
    }, os.path.join(ckpt_root, filename))

    meta = {
        'config': asdict(config),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'eval_metrics': eval_metrics,
        'time_per_epoch': epoch_times[-1],
        'total_time': sum(epoch_times),
        'train_split': TRAIN_SPLIT,
        'val_split': VAL_SPLIT,
        'test_split': TEST_SPLIT
    }
    json.dump(meta, open(os.path.join(ckpt_root, 'metadata.json'), 'w'), indent=2)

    plt.figure()
    x = range(1, len(train_losses)+1)
    plt.plot(x, train_losses, label='train')
    plt.plot(x, val_losses,   label='val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(os.path.join(ckpt_root, 'loss_curve.png')); plt.close()


def _load_model_and_optimizer_for_training(model: Transformer, opt: Optimizer, model_file: str, device: torch.device) -> Tuple[Transformer, Optimizer, int]:
    """
    Loads saved weights and optimizer state from a checkpoint so training can resume seamlessly.

    ## Args:
        model (Transformer): The model instance to load weights into.
        opt (Optimizer): The optimizer instance to load state into.
        model_file (str): Path to the checkpoint file (either a legacy state_dict or a dict with keys
            'model_state', 'optim_state', and 'step_num').
        device (torch.device): Device on which to map the checkpoint (e.g., CPU or GPU).

    ## Returns:
        Tuple[Transformer, Optimizer, int]:
            - model: The same model instance, now with loaded weights.
            - opt: The same optimizer instance, now with loaded state.
            - step_num: Integer step count from the checkpoint (0 if legacy).

    ## Raises:
        FileNotFoundError: If `model_file` does not exist.
        KeyError: If the checkpoint dict is missing required keys ('model_state', 'optim_state', 'step_num').
    """
    # 1. Check file exists
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Checkpoint not found: {model_file}")

    ckpt = torch.load(model_file, map_location=device)

    # 2. Legacy state_dict vs new checkpoint dict
    if not isinstance(ckpt, dict) or 'model_state' not in ckpt:
        # legacy: ckpt is directly the state_dict
        model.load_state_dict(ckpt)
        print(f"Loaded legacy state_dict from {model_file}")
        return model, opt, 0
    else:
        # 3. Verify required keys
        for key in ('model_state', 'optim_state', 'step_num'):
            if key not in ckpt:
                raise KeyError(f"Expected key '{key}' missing in checkpoint: {model_file}")

        model.load_state_dict( ckpt['model_state'] )
        opt.load_state_dict(   ckpt['optim_state'] )
        step_num = ckpt['step_num']
        print(f"Resuming from step {step_num} (loaded from {model_file})")
        return model, opt, step_num


def train_and_evaluate(config: Config, model_file: Optional[str] = None) -> str:
    """
    Trains the Transformer model and evaluates its performance, optionally resuming from a saved checkpoint.

    ## Args:
        config (Config): Configuration object containing all model, data, and training hyperparameters.
        model_file (Optional[str]): Path to a .pt checkpoint file to load and continue training from; if None, training starts from scratch.

    ## Returns:
        str: Path to the directory where this run’s checkpoints, metadata, and plots are saved.

    ## Side effects:
        - Creates a new timestamped checkpoint directory under “checkpoints/”.
        - Saves model state_dict periodically according to config.save_every and at the end.
        - Records and prints per‐epoch train and validation losses.
        - Applies early stopping based on validation loss improvement.
        - After training, runs a test‐set evaluation (including optional metrics) and saves final artifacts.
    """
    # 1. Data
    train_loader, val_loader, test_loader = get_data_loaders(config)

    # 2. Model, optimiser
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(config).to(device)
    opt = optim.Adam(model.parameters(), lr=0, betas=config.optimizer_betas, eps=config.optimizer_eps)
    step_num = 0

    if model_file is not None:
        model, opt, step_num = _load_model_and_optimizer_for_training(model, opt, model_file, device)

    # 3. Training loop
    run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_root = os.path.join('checkpoints', run_stamp)
    os.makedirs(ckpt_root, exist_ok=True)

    train_losses, val_losses, epoch_times = [], [], []

    best_val = float('inf')
    patience_left = config.early_stopping_patience

    for epoch in range(1, config.num_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        running_loss, n_batches = 0.0, 0

        for src, tgt_in, tgt_lbl in train_loader:
            # warm-up LR schedule
            step_num += 1
            lr_scale = _calculate_lr_scale(step_num, config.d_model, config.warmup_steps)
            opt.param_groups[0]['lr'] = config.lr * lr_scale

            src, tgt_in, tgt_lbl = src.to(device), tgt_in.to(device), tgt_lbl.to(device)
            opt.zero_grad()
            loss = model(src, tgt_in, labels=tgt_lbl)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            n_batches += 1

        avg_train = running_loss / n_batches
        train_losses.append(avg_train)

        model.eval()
        running_val, n_val_batches = 0.0, 0

        with torch.no_grad():
            for src, tgt_in, tgt_lbl in val_loader:
                src, tgt_in, tgt_lbl = src.to(device), tgt_in.to(device), tgt_lbl.to(device)
                running_val += model(src, tgt_in, labels=tgt_lbl).item()
                n_val_batches += 1

        avg_val = running_val / n_val_batches
        val_losses.append(avg_val)

        epoch_times.append(time.perf_counter() - t0)
        print(f'Epoch {epoch:02d}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}')

        if epoch % config.save_every == 0 or epoch == config.num_epochs:
            _save(config, model, opt, step_num, ckpt_root, f'model_epoch{epoch}.pt', epoch_times, train_losses, val_losses)

        if best_val - avg_val > config.early_stopping_min_delta:
            best_val = avg_val
            patience_left = config.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f'Early stopping triggered at epoch {epoch}.')
                break

    # 4. Test
    try:
        metrics = test_model(model, test_loader, tokenizer, device, config)
        print(f'Metrics: {metrics}')
        _save(config, model, opt, step_num, ckpt_root, 'model_final.pt', epoch_times, train_losses, val_losses, metrics)
    except Exception as e:
        print('Error:', e)
        _save(config, model, opt, step_num, ckpt_root, 'model_final.pt', epoch_times, train_losses, val_losses)

    return ckpt_root


def test_model(model: Transformer, test_loader: DataLoader, tokenizer: AutoTokenizer, device: device, config: Config) -> Dict[str, float]:
    """
    Evaluates the Transformer model on the test set and computes a suite of translation metrics.

    ## Args:
        model (Transformer): The trained Transformer model to evaluate.
        test_loader (DataLoader): DataLoader yielding batches of (src, tgt_in, tgt_lbl) for the test split.
        tokenizer (AutoTokenizer): Tokenizer used to decode model outputs and targets into text.
        device (torch.device): The device (CPU or CUDA) on which to perform inference.
        config (Config): Configuration object containing generation parameters (e.g., max output length, beam size).

    ## Returns:
        Dict[str, float]: A dictionary containing:
            - test_loss (float): Average cross-entropy loss over the test set.
            - bleu (float): SacreBLEU score (0–100).
            - chrf (float): ChrF score (0–100).
            - rouge1 (float): ROUGE-1 F1 score.
            - rouge2 (float): ROUGE-2 F1 score.
            - rougeL (float): ROUGE-L F1 score.
            - meteor (float): METEOR score.
            - ter (float): Translation Error Rate (TER).
            - bertscore_f1 (float): BERTScore F1.

    ## Side effects:
        - Runs model.generate(...) on each batch to produce translations.
        - Accumulates predictions and references for all metrics.
    """
    # load metrics
    bleu   = evaluate.load('sacrebleu')
    chrf   = evaluate.load('chrf')
    rouge  = evaluate.load('rouge')
    meteor = evaluate.load('meteor')
    ter    = evaluate.load('ter')
    bert   = evaluate.load('bertscore')

    model.eval()
    running_loss, n_batches = 0.0, 0
    all_preds, all_refs = [], []

    with torch.no_grad():
        for src, tgt_in, tgt_lbl in tqdm(test_loader, desc='Testing', total=len(test_loader), leave=False):
            src, tgt_in, tgt_lbl = src.to(device), tgt_in.to(device), tgt_lbl.to(device)

            # loss
            loss = model(src, tgt_in, labels=tgt_lbl)
            running_loss += loss.item()
            n_batches += 1

            # generate
            gen_ids = model.generate(
                src,
                max_length=config.output_seq_len,
                num_beams=4,
                early_stopping=True
            )
            preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            refs  = tokenizer.batch_decode(tgt_lbl, skip_special_tokens=True)

            all_preds.extend(preds)
            all_refs.extend([[r] for r in refs])  # list of lists

    avg_loss = running_loss / n_batches

    # compute metrics
    out = {}
    out['test_loss'] = avg_loss
    out['bleu'] = bleu.compute(predictions=all_preds, references=all_refs)['score']
    out['chrf'] = chrf.compute(predictions=all_preds, references=all_refs)['score']
    out.update(rouge.compute(predictions=all_preds, references=[r[0] for r in all_refs], use_stemmer=True))
    out['meteor'] = meteor.compute(predictions=all_preds, references=[r[0] for r in all_refs])['meteor']
    out['ter'] = ter.compute(predictions=all_preds, references=[r[0] for r in all_refs])['score']
    bert_res = bert.compute(predictions=all_preds, references=[r[0] for r in all_refs], lang='en')
    out['bertscore_f1'] = sum(bert_res['f1']) / len(bert_res['f1'])

    return out


def load_model(ckpt_dir: str, model_file: str = 'model_final.pt') -> Transformer:
    """
    Loads a saved Transformer model from the specified checkpoint directory.

    ## Args:
        ckpt_dir (str): Path to the checkpoint directory containing:
            ├─ model.pt or model_epoch{N}.pt (legacy state_dict)
            ├─ model_final.pt or any checkpoint dict with keys 'model_state', 'optim_state', 'step_num'
            ├─ loss_curve.png
            └─ metadata.json
        model_file (str): Name of the checkpoint file to load (default: "model_final.pt").

    ## Returns:
        Transformer: A Transformer instance initialized with the loaded state_dict,
                     moved to the appropriate device, and set to evaluation mode.
    """
    # 1. Load metadata to reconstruct config
    meta_path = os.path.join(ckpt_dir, 'metadata.json')
    meta = json.load(open(meta_path, 'r'))
    config = Config(**meta['config'])

    # 2. Instantiate model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(config).to(device)

    # 3. Load checkpoint
    ckpt_path = os.path.join(ckpt_dir, model_file)
    ckpt = torch.load(ckpt_path, map_location=device)

    # 4. Handle both new dict-style and legacy state_dict
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        state_dict = ckpt['model_state']
    else:
        state_dict = ckpt

    # 5. Load weights & set eval mode
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.inference_mode()
def translate(
    model: Transformer,
    sentence: str,
    max_len: int = 100,
    beam_width: int = 5,
    len_penalty: float = 0.7
) -> str:
    """
    Beam-search translate an English `sentence` → Czech string.

    ## Args:
        model (Transformer): A Transformer returned by `load_model`, already in eval mode.
        sentence (str): Raw English input text to translate.
        max_len (int): Generation cap (including special BOS/EOS tokens).
        beam_width (int): Number of beams to keep during search.
        len_penalty (float): Length penalty >0 to penalize short sequences (0.6–1.0 typical).

    ## Returns:
        str: The decoded Czech translation.

    ## Side effects:
        - Tokenizes the input sentence.
        - Initializes the beam search with the BOS token.
        - Iteratively expands beams by running the model's decoder.
        - Applies length penalty and stops when EOS is generated or max_len is reached.
        - Decodes the best beam's token IDs back into a human-readable string.

    ## Example:
        >>> model = load_model('checkpoints/20250705_123456/model_final.pt')
        >>> translate(model, "Good morning!", beam_width=4, len_penalty=0.8)
        "Dobré ráno!"
    """
    # special token IDs
    BOS_ID_SRC = vocab_src['<bos>']
    EOS_ID_SRC = vocab_src['<eos>']
    EOS_ID_TGT = vocab_tgt['<eos>']

    # prepare source tensor
    device = next(model.parameters()).device
    src_tok = tokenizer.tokenize(sentence.lower().strip())
    src_ids = [BOS_ID_SRC] + [vocab_src[t] for t in src_tok] + [EOS_ID_SRC]
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T_src)

    # delegate beam-search to model.generate()
    out_ids = model.generate(
        src,
        max_length=max_len,
        num_beams=beam_width,
        length_penalty=len_penalty
    )

    # extract best sequence from tensor, drop BOS/EOS
    seq = out_ids[0].tolist()
    if EOS_ID_TGT in seq:
        seq = seq[1:seq.index(EOS_ID_TGT)]
    else:
        seq = seq[1:]

    # decode token IDs → text
    tokens_cs = [vocab_tgt.lookup_token(i) for i in seq]
    output = ' '.join(tokens_cs).replace('▁', '')
    return output