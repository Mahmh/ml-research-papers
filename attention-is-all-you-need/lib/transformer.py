from typing import Optional
from torch import Tensor, LongTensor
import math, torch, torch.nn as nn, torch.nn.functional as F
from lib.constants import Config

class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        max_seq_len = max(config.input_seq_len, config.output_seq_len)
        self.config = config
        self.src_embeddings = nn.Embedding(config.src_vocab_size, config.d_model, padding_idx=config.src_pad_idx)
        self.tgt_embeddings = nn.Embedding(config.tgt_vocab_size, config.d_model, padding_idx=config.tgt_pad_idx)
        self.positional_embeddings = nn.Parameter(torch.randn(max_seq_len, config.d_model))
        self.encoders = nn.ModuleList([Encoder(config) for _ in range(config.num_encoders)])
        self.decoders = nn.ModuleList([Decoder(config) for _ in range(config.num_decoders)])
        self.final_fc = nn.Linear(config.d_model, config.tgt_vocab_size)

    def forward(self, source_batch: Tensor, target_batch: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        """
        ### Input Shapes
        - `source_batch`: (batch_size, input_seq_len)
        - `target_batch`: (batch_size, output_seq_len)
        - `labels`: (batch_size, output_seq_len)
        ### Output Shape
        - returns:
            - If `labels` is provided: *the scalar cross-entropy loss*
            - Else: *output logits* (batch_size, output_seq_len, tgt_vocab_size)
        """
        # 1. Embed source
        src_emb = self.src_embeddings(source_batch) + self.positional_embeddings[:source_batch.size(1)].unsqueeze(0)  # (batch_size, input_seq_len, d_model)

        # 2. Encoder stack
        enc_out = src_emb
        for encoder in self.encoders:
            enc_out = encoder(enc_out)  # (batch_size, input_seq_len, d_model)

        # 3. Embed target (shifted right outside this function)
        tgt_emb = self.tgt_embeddings(target_batch) + self.positional_embeddings[:target_batch.size(1)].unsqueeze(0)  # (batch_size, output_seq_len, d_model)

        # 4. Decoder stack (with cross-attention to encoder outputs)
        dec_in = tgt_emb
        for decoder in self.decoders:
            dec_in = decoder(dec_in, enc_out)  # (batch_size, output_seq_len, d_model)
        x = dec_in  # (batch_size, output_seq_len, d_model)

        # 5. Final linear layer -> vocabulary logits
        logits = self.final_fc(x)  # (batch_size, output_seq_len, tgt_vocab_size)

        # 6. Compute loss or return logits
        if labels is not None:
            loss = F.cross_entropy(
                logits.transpose(1, 2),  # (batch_size, tgt_vocab_size, output_seq_len)
                labels,                  # (batch_size, output_seq_len)
                ignore_index=self.config.tgt_pad_idx
            )
            return loss  # scalar tensor
        else:
            return logits
    

    @torch.no_grad()
    def generate(self, src: LongTensor, max_length: int = 100, num_beams: int = 1, length_penalty: float = 0.7, early_stopping: bool = True) -> LongTensor:
        """
        Beam-search decode (falls back to greedy when `num_beams=1`).

        ## Args:
            src (LongTensor): (B, T_src) source token IDs with BOS/EOS.
            max_length (int): maximum total output length (incl. BOS/EOS).
            num_beams (int): number of beams to keep at each step.
            length_penalty (float): >0 to penalize short sequences (0.6–1.0 typical).
            early_stopping (bool): if True, stop when all beams end.

        ## Returns:
            LongTensor: (B, L_out) best token ID sequences per batch element.
        """
        B, device = src.size(0), src.device

        # 1) encode once
        enc = self.src_embeddings(src) + self.positional_embeddings[: src.size(1)].unsqueeze(0)
        for layer in self.encoders:
            enc = layer(enc)

        # 2) init beams
        beams = [[([self.config.tgt_bos_idx], 0.0)] for _ in range(B)]

        for _ in range(max_length - 1):
            new_beams = []
            for b in range(B):
                candidates = []
                for seq, score in beams[b]:
                    if seq[-1] == self.config.tgt_eos_idx:
                        candidates.append((seq, score))
                        continue

                    tgt = torch.tensor(seq, device=device).unsqueeze(0)
                    dec = self.tgt_embeddings(tgt) + self.positional_embeddings[:tgt.size(1)].unsqueeze(0)
                    for layer in self.decoders:
                        dec = layer(dec, enc[b : b+1])
                    logp = F.log_softmax(self.final_fc(dec[:, -1, :]), dim=-1).squeeze(0)

                    topv, topi = logp.topk(num_beams)
                    for tok_score, tok_id in zip(topv.tolist(), topi.tolist()):
                        candidates.append((seq + [tok_id], score + tok_score))

                # length‐penalised sort & prune
                candidates = sorted(
                    candidates,
                    key=lambda x: x[1] / (len(x[0]) ** length_penalty),
                    reverse=True
                )[:num_beams]
                new_beams.append(candidates)

            beams = new_beams

            # break only if early_stopping is True
            if early_stopping and all(
                seq[-1] == self.config.tgt_eos_idx
                for beam_list in beams
                for seq, _ in beam_list
            ):
                break

        # pick best and pad
        output, max_len = [], 0
        for beam_list in beams:
            seq, _ = beam_list[0]
            seq = seq[1 : seq.index(self.config.tgt_eos_idx)] if self.config.tgt_eos_idx in seq else seq[1:]
            output.append(seq); max_len = max(max_len, len(seq))

        out_tensor = torch.full((B, max_len), self.config.tgt_pad_idx, dtype=torch.long, device=device)
        for i, seq in enumerate(output):
            out_tensor[i, :len(seq)] = torch.tensor(seq, device=device)

        return out_tensor


class Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(config) for _ in range(config.num_heads)])
        self.W_O = nn.Linear(config.d_model, config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),  # (batch_size, input_seq_len, d_ff)
            nn.ReLU(),
            nn.Linear(config.d_ff,  config.d_model)  # (batch_size, input_seq_len, d_model)
        )
        self.dropout1 = nn.Dropout(config.p_drop)
        self.dropout2 = nn.Dropout(config.p_drop)
        self.layernorm1 = nn.LayerNorm(config.d_model)
        self.layernorm2 = nn.LayerNorm(config.d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        ### Input Shapes
        - `x`: (batch_size, input_seq_len, d_model)
        ### Output Shape
        - returns: (batch_size, input_seq_len, d_model)
        """
        head_outputs = [head(x) for head in self.heads]  # each head output: (batch_size, input_seq_len, d_k)
        head_outputs = torch.cat(head_outputs, dim=-1)  # (batch_size, input_seq_len, d_model)
        multihead_attn = self.W_O(head_outputs)  # (batch_size, input_seq_len, d_model)

        x1 = self.layernorm1(x + self.dropout1(multihead_attn))  # (batch_size, input_seq_len, d_model)
        ffn_out = self.ffn(x1)  # (batch_size, input_seq_len, d_model)
        x2 = self.layernorm2(x1 + self.dropout2(ffn_out))  # (batch_size, input_seq_len, d_model)
        return x2


class Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(config) for _ in range(config.num_heads)])
        self.W_O = nn.Linear(config.d_model, config.d_model)
        self.mmha_heads = nn.ModuleList([CrossAttentionHead(config) for _ in range(config.num_heads)])
        self.mmha_W_O = nn.Linear(config.d_model, config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),  # (batch_size, output_seq_len, d_ff)
            nn.ReLU(),
            nn.Linear(config.d_ff,  config.d_model)  # (batch_size, output_seq_len, d_model)
        )
        self.dropout1 = nn.Dropout(config.p_drop)
        self.dropout2 = nn.Dropout(config.p_drop)
        self.dropout3 = nn.Dropout(config.p_drop)
        self.layernorm1 = nn.LayerNorm(config.d_model)
        self.layernorm2 = nn.LayerNorm(config.d_model)
        self.layernorm3 = nn.LayerNorm(config.d_model)

    def forward(self, dec_input: Tensor, enc_output: Tensor) -> Tensor:
        """
        ### Input Shapes
        - `dec_input`:  (batch_size, output_seq_len, d_model)
        - `enc_output`: (batch_size, input_seq_len,  d_model)
        ### Output Shape
        - returns:      (batch_size, output_seq_len, d_model)
        """
        B, T_out, _ = dec_input.shape  # batch_size & output_seq_len can be different during inference

        # 1. Masked Self-Attention
        # Create causal mask to allow attending to current and past positions
        mask = torch.tril(torch.ones(T_out, T_out, device=dec_input.device, dtype=torch.bool))  # (batch_size, output_seq_len, output_seq_len)
        mask = mask.unsqueeze(0).expand(B, T_out, T_out)
        # Project and attend per head
        self_attn_heads = [head(dec_input, mask) for head in self.heads]  # each head output: (batch_size, output_seq_len, d_k)
        self_attn = torch.cat(self_attn_heads, dim=-1)  # (batch_size, output_seq_len, d_model)
        self_attn = self.W_O(self_attn)  # (batch_size, output_seq_len, d_model)
        x1 = self.layernorm1(dec_input + self.dropout1(self_attn))  # (batch_size, output_seq_len, d_model)

        # 2. Encoder-Decoder Cross-Attention
        cross_heads = [head(x1, enc_output, mask=None) for head in self.mmha_heads] #  each head output: (batch_size, output_seq_len, d_k)
        cross_attn = torch.cat(cross_heads, dim=-1)  # (batch_size, output_seq_len, d_model)
        cross_attn = self.mmha_W_O(cross_attn)  # (batch_size, output_seq_len, d_model)
        x2 = self.layernorm2(x1 + self.dropout2(cross_attn)) #  (batch_size, output_seq_len, d_model)

        # 3. Position-wise Feed-Forward
        ffn_out = self.ffn(x2)  # (batch_size, output_seq_len, d_model)
        x3 = self.layernorm3(x2 + self.dropout3(ffn_out))  # (batch_size, output_seq_len, d_model)
        return x3



class SelfAttentionHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.d_k = config.d_model // config.num_heads
        self.W_Q = nn.Linear(config.d_model, self.d_k)
        self.W_K = nn.Linear(config.d_model, self.d_k)
        self.W_V = nn.Linear(config.d_model, self.d_k)
        self.dropout = nn.Dropout(config.p_drop)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        ### Input Shapes
        - `x`: (batch_size, input_seq_len, d_model)
        - `mask`: (batch_size, input_seq_len, input_seq_len)  # 1 = keep, 0 = mask
        ### Output Shape
        - returns: (batch_size, input_seq_len, d_k)
        """
        # 1. Project
        Q = self.W_Q(x)  # (batch_size, input_seq_len, d_k)
        K = self.W_K(x)  # (batch_size, input_seq_len, d_k)
        V = self.W_V(x)  # (batch_size, input_seq_len, d_k)

        # 2. Compute scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, input_seq_len, input_seq_len)
        scores = scores / math.sqrt(self.d_k)  # (batch_size, input_seq_len, input_seq_len)

        # 3. Apply mask (if provided) so masked positions (-inf) are converted to 0s via softmax
        if mask is not None:  scores = scores.masked_fill(mask == 0, float('-inf'))

        # 4. Normalize into probabilities
        attn = F.softmax(scores, dim=-1)  # (batch_size, input_seq_len, input_seq_len)
        attn = self.dropout(attn)

        # 5. Apply to values
        out = torch.matmul(attn, V)  # (batch_size, input_seq_len, d_k)
        return out


class CrossAttentionHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.d_k = config.d_model // config.num_heads
        self.W_Q = nn.Linear(config.d_model, self.d_k)
        self.W_K = nn.Linear(config.d_model, self.d_k)
        self.W_V = nn.Linear(config.d_model, self.d_k)
        self.dropout = nn.Dropout(config.p_drop)
    
    def forward(self, dec_x: Tensor, enc_x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        ### Input Shapes
        - `dec_x`: (batch_size, output_seq_len, d_model)
        - `enc_x`: (batch_size, input_seq_len,  d_model)
        - `mask`:  (batch_size, output_seq_len, input_seq_len)  # 1 = keep, 0 = mask
        ### Output Shape
        - returns: (batch_size, output_seq_len, d_k)
        """
        # 1. Project queries from decoder input, keys/values from encoder output
        Q = self.W_Q(dec_x)   # (batch_size, output_seq_len, d_k)
        K = self.W_K(enc_x)   # (batch_size, input_seq_len, d_k)
        V = self.W_V(enc_x)   # (batch_size, input_seq_len, d_k)

        # 2. Compute scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))  #(batch_size, output_seq_len, input_seq_len)
        scores = scores / math.sqrt(self.d_k)

        # 3. Apply mask (if provided)
        if mask is not None:  scores = scores.masked_fill(mask == 0, float('-inf'))

        # 4. Normalize into probabilities
        attn = F.softmax(scores, dim=-1)  # (batch_size, output_seq_len, input_seq_len)
        attn = self.dropout(attn)

        # 5. Weight and sum values
        out = torch.matmul(attn, V) # (batch_size, output_seq_len, d_k)
        return out