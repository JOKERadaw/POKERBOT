"""
poker_model.py
==============
Encoder-only Transformer for 6-max No Limit Texas Hold'em.

Architecture summary
--------------------
  Input tokens (sequence of 64, always fixed-length):
    [CLS] | 2 hole-card | 5 community-card | pos | street | pot | hero_stk | vill_stk | 51 action
     0         1-2            3-7             8      9      10      11         12         13-63

  Each token type has its own embedding → all projected to d_model = 128.
  A learnable positional embedding is added to every position.

  Four encoder layers (pre-LN, 4 heads, d_ff=256) process the sequence.

  The [CLS] representation at position 0 is fed into a shared MLP trunk:
    Linear(128 → 64) → ReLU
  which branches into:
    - action_head : Linear(64 → 4)    → softmax at inference
    - amount_head : Linear(64 → 1)    → sigmoid → scale ×2.0  (maps to 0-2× pot)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from poker_game_state import (
    CARD_VOCAB,      # 53
    ACTION_VOCAB,    # 6
    POSITION_VOCAB,  # 6
    STREET_VOCAB,    # 4
    NULL_CARD,
    ACTION_PAD,
    MAX_SEQ_LEN,     # 64
    MAX_ACTIONS,     # 51
    PREFIX_LEN,      # 13
)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 4
D_FF     = 256
DROPOUT  = 0.1
N_OUT    = 4    # fold | check/call | bet | raise


class PokerTransformer(nn.Module):
    """
    Encoder-only Transformer that maps a structured poker game state to
    (action_logits, amount_pred).

    Parameters
    ----------
    d_model  : token embedding dimension (default 128)
    n_heads  : number of attention heads  (default 4)
    n_layers : number of encoder layers   (default 4)
    d_ff     : feed-forward hidden dim    (default 256)
    dropout  : dropout probability        (default 0.1)
    """

    def __init__(
        self,
        d_model:  int   = D_MODEL,
        n_heads:  int   = N_HEADS,
        n_layers: int   = N_LAYERS,
        d_ff:     int   = D_FF,
        dropout:  float = DROPOUT,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # ── [CLS] learnable token ────────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))

        # ── Card token embedding ─────────────────────────────────────────────
        # Vocab: 0-51 = cards,  52 = NULL (undealt community card)
        self.card_emb = nn.Embedding(CARD_VOCAB, d_model)

        # ── Discrete context embeddings ──────────────────────────────────────
        self.position_emb = nn.Embedding(POSITION_VOCAB, d_model)  # seat 0-5
        self.street_emb   = nn.Embedding(STREET_VOCAB,   d_model)  # 0-3

        # ── Scalar context projection ────────────────────────────────────────
        # The three continuous scalars (pot, hero_stack, villain_stack) each
        # become one token via a shared linear layer R^1 → R^d_model.
        self.scalar_proj = nn.Linear(1, d_model)

        # ── Action token embedding ────────────────────────────────────────────
        # Each action token = (type_id, normalised_amount).
        # Type  → Embedding(ACTION_VOCAB, d_model//2)
        # Amount→ Linear(1, d_model//2)
        # Fused → Linear(d_model, d_model)   with bias + ReLU
        self.act_type_emb    = nn.Embedding(ACTION_VOCAB, d_model // 2)
        self.act_amount_proj = nn.Linear(1, d_model // 2, bias=False)
        self.act_fuse        = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        # ── Learnable positional encoding ────────────────────────────────────
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, d_model)

        # ── Transformer encoder ───────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_ff,
            dropout         = dropout,
            batch_first     = True,   # (B, S, d_model) convention
            norm_first      = True,   # pre-LN for stable training
        )
        # enable_nested_tensor=False: pre-LN cannot use the nested-tensor fast path;
        # disabling it here suppresses a spurious PyTorch UserWarning.
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # ── Dual-output MLP head ─────────────────────────────────────────────
        self.mlp_trunk   = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU())
        self.action_head = nn.Linear(64, N_OUT)     # logits for 4 action classes
        self.amount_head = nn.Sequential(            # sigmoid → ×2 = 0..2× pot
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    # ── Initialisation ────────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ── Sequence builder ──────────────────────────────────────────────────────
    def _build_sequence(
        self,
        card_ids:    torch.Tensor,   # (B, 7)
        ctx_disc:    torch.Tensor,   # (B, 2)
        ctx_scalar:  torch.Tensor,   # (B, 3)
        act_types:   torch.Tensor,   # (B, 51)
        act_amounts: torch.Tensor,   # (B, 51)
    ) -> torch.Tensor:               # (B, 64, d_model)
        B = card_ids.size(0)

        # --- token 0 : CLS ---------------------------------------------------
        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, d_model)

        # --- tokens 1-2 : hole cards, 3-7 : community cards ------------------
        card_embs = self.card_emb(card_ids)                 # (B, 7, d_model)

        # --- token 8 : position, token 9 : street ----------------------------
        pos_tok    = self.position_emb(ctx_disc[:, 0]).unsqueeze(1)  # (B,1,d)
        street_tok = self.street_emb(ctx_disc[:, 1]).unsqueeze(1)    # (B,1,d)

        # --- tokens 10-12 : pot / hero_stack / villain_stack -----------------
        # ctx_scalar: (B,3) → (B,3,1) → (B,3,d_model)
        scalar_toks = self.scalar_proj(ctx_scalar.unsqueeze(-1))     # (B,3,d)

        # --- tokens 13-63 : action sequence ----------------------------------
        a_type   = self.act_type_emb(act_types)                      # (B,51,d//2)
        a_amount = self.act_amount_proj(act_amounts.unsqueeze(-1))   # (B,51,d//2)
        a_fused  = self.act_fuse(torch.cat([a_type, a_amount], dim=-1))  # (B,51,d)

        # --- concatenate all token embeddings into one sequence --------------
        # Shape: (B, 1+7+1+1+3+51, d_model) = (B, 64, d_model)
        seq = torch.cat(
            [cls, card_embs, pos_tok, street_tok, scalar_toks, a_fused],
            dim=1,
        )

        # --- add learnable positional encodings ------------------------------
        positions = torch.arange(MAX_SEQ_LEN, device=seq.device).unsqueeze(0)
        seq = seq + self.pos_emb(positions)   # (B, 64, d_model)

        return seq

    # ── Padding mask builder ──────────────────────────────────────────────────
    @staticmethod
    def _build_pad_mask(act_mask: torch.Tensor) -> torch.Tensor:
        """
        Build the src_key_padding_mask for nn.TransformerEncoder.
        Shape: (B, 64)  — True where the position should be *ignored*.
        The first PREFIX_LEN=13 tokens are structural; they are never masked.
        Action slots where act_mask is False (padding) are masked out.
        """
        B = act_mask.size(0)
        device = act_mask.device
        prefix_ok = torch.zeros(B, PREFIX_LEN, dtype=torch.bool, device=device)
        act_pad   = ~act_mask   # True = padding → should be ignored
        return torch.cat([prefix_ok, act_pad], dim=1)                # (B, 64)

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(
        self,
        card_ids:    torch.Tensor,   # (B, 7)   LongTensor
        ctx_disc:    torch.Tensor,   # (B, 2)   LongTensor  [pos, street]
        ctx_scalar:  torch.Tensor,   # (B, 3)   FloatTensor [pot_n, hero_n, vill_n]
        act_types:   torch.Tensor,   # (B, 51)  LongTensor
        act_amounts: torch.Tensor,   # (B, 51)  FloatTensor
        act_mask:    torch.Tensor,   # (B, 51)  BoolTensor  True = valid action
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        action_logits : (B, 4)  raw logits for [fold, check/call, bet, raise]
        amount_pred   : (B, 1)  bet size as fraction of pot in range (0, 2.0)
        """
        # Build the full 64-token input sequence
        seq = self._build_sequence(card_ids, ctx_disc, ctx_scalar, act_types, act_amounts)

        # Padding mask: shape (B, 64), True = ignore
        pad_mask = self._build_pad_mask(act_mask)

        # Transformer encoder: (B, 64, d_model) → (B, 64, d_model)
        encoded = self.transformer(seq, src_key_padding_mask=pad_mask)

        # Pool via [CLS] token (position 0)
        cls_repr = encoded[:, 0, :]          # (B, d_model)

        # Shared MLP trunk
        trunk = self.mlp_trunk(cls_repr)     # (B, 64)

        # Action logits
        action_logits = self.action_head(trunk)            # (B, 4)

        # Amount prediction: sigmoid in (0,1) × 2 → (0, 2.0) pot fractions
        amount_pred = self.amount_head(trunk) * 2.0        # (B, 1)

        return action_logits, amount_pred

    # ── Convenience: predict a single GameState ───────────────────────────────
    @torch.no_grad()
    def predict(self, sample: dict, device: torch.device | str = "cpu"):
        """
        Run inference on a single tokenize() output (no batch dimension).
        Returns (action_name, amount_fraction, action_logits, amount_scalar).
        """
        from poker_game_state import OUTPUT_ACTION_NAMES

        self.eval()
        batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()
                 if k not in ("label_action", "label_amount")}

        logits, amount = self(
            batch["card_ids"],
            batch["ctx_disc"],
            batch["ctx_scalar"],
            batch["act_types"],
            batch["act_amounts"],
            batch["act_mask"],
        )

        action_idx   = logits.argmax(dim=-1).item()
        action_name  = OUTPUT_ACTION_NAMES[action_idx]
        amount_val   = amount.item()   # fraction of pot

        return action_name, amount_val, logits.squeeze(0), amount.squeeze(0)


def build_poker_transformer(**kwargs) -> PokerTransformer:
    """Factory function — pass any hyper-parameter overrides as kwargs."""
    model = PokerTransformer(**kwargs)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[PokerTransformer] Parameters: {total:,}")
    return model
