"""
poker_game_state.py
===================
Defines the GameState dataclass and the tokenizer that converts a GameState
into the fixed-length tensor representation consumed by PokerTransformer.

Sequence layout (always exactly MAX_SEQ_LEN = 64 tokens):
  [CLS] | hole_0 hole_1 | comm_0..comm_4 | ctx_pos ctx_street | ctx_pot ctx_hero ctx_vill | a_0..a_50
   pos0       1-2             3-7              8       9           10       11       12       13-63

Token counts: 1 + 2 + 5 + 2 + 3 + 51 = 64
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import torch

# ── Card encoding ──────────────────────────────────────────────────────────────
# Ranks: 2=0 … A=12    Suits: c=0, d=1, h=2, s=3
# card_id = rank_index * 4 + suit_index  →  range 0..51
# NULL_CARD = 52  (undealt community card placeholder)

RANK_CHARS: str = "23456789TJQKA"
SUIT_CHARS: str = "cdhs"

RANK_MAP: Dict[str, int] = {r: i for i, r in enumerate(RANK_CHARS)}
SUIT_MAP: Dict[str, int] = {s: i for i, s in enumerate(SUIT_CHARS)}
RANK_INV: Dict[int, str] = {v: k for k, v in RANK_MAP.items()}
SUIT_INV: Dict[int, str] = {v: k for k, v in SUIT_MAP.items()}

NULL_CARD: int = 52   # padding for undealt community cards
CARD_VOCAB: int = 53  # 0-51 cards + 52 NULL


def card_to_id(card_str: str) -> int:
    """Convert a card string like 'Kd' or 'ah' to an integer ID (0-51)."""
    rank = RANK_MAP[card_str[0].upper()]
    suit = SUIT_MAP[card_str[1].lower()]
    return rank * 4 + suit


def id_to_card(card_id: int) -> str:
    """Convert an integer ID back to a card string like 'Kd'."""
    return RANK_INV[card_id // 4] + SUIT_INV[card_id % 4]


# ── Action encoding ────────────────────────────────────────────────────────────
FOLD:       int = 0
CHECK:      int = 1
CALL:       int = 2
BET:        int = 3
RAISE:      int = 4
ACTION_PAD: int = 5   # padding token for unused action slots

ACTION_VOCAB: int = 6  # 0-4 real actions + 5 PAD

ACTION_NAMES = {0: "fold", 1: "check", 2: "call", 3: "bet", 4: "raise"}

# Output head classes (model predicts one of these four)
OUT_FOLD:       int = 0
OUT_CHECK_CALL: int = 1
OUT_BET:        int = 2
OUT_RAISE:      int = 3

OUTPUT_ACTION_NAMES = {0: "fold", 1: "check/call", 2: "bet", 3: "raise"}

# Map raw action → output action class
ACTION_TO_OUTPUT = {
    FOLD:  OUT_FOLD,
    CHECK: OUT_CHECK_CALL,
    CALL:  OUT_CHECK_CALL,
    BET:   OUT_BET,
    RAISE: OUT_RAISE,
}

# ── Position encoding ──────────────────────────────────────────────────────────
POSITIONS = {"UTG": 0, "HJ": 1, "CO": 2, "BTN": 3, "SB": 4, "BB": 5}
POSITION_INV = {v: k for k, v in POSITIONS.items()}
POSITION_VOCAB: int = 6

# ── Street encoding ────────────────────────────────────────────────────────────
PREFLOP: int = 0
FLOP:    int = 1
TURN:    int = 2
RIVER:   int = 3
STREET_VOCAB: int = 4

# ── Sequence layout constants ──────────────────────────────────────────────────
MAX_SEQ_LEN:   int = 64
# Non-action prefix: CLS(1) + hole(2) + comm(5) + pos(1) + street(1) + scalars(3) = 13
PREFIX_LEN:    int = 13
MAX_ACTIONS:   int = MAX_SEQ_LEN - PREFIX_LEN   # 51


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class PokerAction:
    """A single in-game action by any player."""
    action_type: int    # FOLD/CHECK/CALL/BET/RAISE  (0-4)
    amount:      float  # raw chip amount (0 for check/fold)


@dataclass
class GameState:
    """
    Complete description of a poker decision point for the hero.

    Attributes
    ----------
    hole_cards       : list of exactly 2 card IDs (0-51)
    community_cards  : list of 0-5 card IDs (undealt are NOT added here;
                       the tokenizer pads with NULL_CARD to exactly 5)
    actions          : chronological list of PokerActions *preceding* the
                       hero's current decision (all streets combined)
    position         : hero's seat index (0=UTG, 1=HJ, 2=CO, 3=BTN, 4=SB, 5=BB)
    pot              : total pot size in chips at the decision point
    hero_stack       : hero's remaining stack
    villain_stack    : representative villain stack (average or main villain)
    street           : current street (0=preflop … 3=river)
    starting_stack   : initial stack size used for normalisation (default 100)
    label_action     : ground-truth output class 0-3 (optional; for training)
    label_amount     : ground-truth bet fraction of pot 0-2 (optional; for training)
    """
    hole_cards:      List[int]
    community_cards: List[int]
    actions:         List[PokerAction]
    position:        int
    pot:             float
    hero_stack:      float
    villain_stack:   float
    street:          int
    starting_stack:  float          = 100.0
    label_action:    Optional[int]  = None
    label_amount:    Optional[float]= None


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def tokenize(gs: GameState) -> Dict[str, torch.Tensor]:
    """
    Convert a GameState into a dict of tensors (one sample, no batch dim).

    Returns
    -------
    card_ids    : LongTensor   [7]           hole (2) + community (5, NULL-padded)
    ctx_disc    : LongTensor   [2]           [position_id, street_id]
    ctx_scalar  : FloatTensor  [3]           [pot/S, hero_stack/S, villain_stack/S]
    act_types   : LongTensor   [MAX_ACTIONS] action type ids, ACTION_PAD where unused
    act_amounts : FloatTensor  [MAX_ACTIONS] normalised amounts, 0.0 where unused
    act_mask    : BoolTensor   [MAX_ACTIONS] True = valid action (not padding)
    label_action: LongTensor   []            scalar class label  (−1 if absent)
    label_amount: FloatTensor  []            scalar amount label (0  if absent)
    """
    S = gs.starting_stack

    # 1. Card tokens ─────────────────────────────────────────────────────────
    community_padded = gs.community_cards + [NULL_CARD] * (5 - len(gs.community_cards))
    card_ids = torch.tensor(gs.hole_cards + community_padded, dtype=torch.long)  # [7]

    # 2. Discrete context ────────────────────────────────────────────────────
    ctx_disc = torch.tensor([gs.position, gs.street], dtype=torch.long)  # [2]

    # 3. Scalar context (normalised to [0,1] roughly) ────────────────────────
    ctx_scalar = torch.tensor(
        [gs.pot / S, gs.hero_stack / S, gs.villain_stack / S],
        dtype=torch.float,
    )  # [3]

    # 4. Action tokens ───────────────────────────────────────────────────────
    # Truncate to MAX_ACTIONS; older actions are dropped from the front if needed
    actions = gs.actions[-MAX_ACTIONS:]
    n_act = len(actions)

    act_types   = torch.full((MAX_ACTIONS,), ACTION_PAD, dtype=torch.long)
    act_amounts = torch.zeros(MAX_ACTIONS, dtype=torch.float)
    act_mask    = torch.zeros(MAX_ACTIONS, dtype=torch.bool)

    for i, a in enumerate(actions):
        act_types[i]   = a.action_type
        act_amounts[i] = a.amount / S
        act_mask[i]    = True

    # 5. Labels ──────────────────────────────────────────────────────────────
    lbl_action = torch.tensor(
        gs.label_action if gs.label_action is not None else -1,
        dtype=torch.long,
    )
    lbl_amount = torch.tensor(
        gs.label_amount if gs.label_amount is not None else 0.0,
        dtype=torch.float,
    )

    return {
        "card_ids":     card_ids,
        "ctx_disc":     ctx_disc,
        "ctx_scalar":   ctx_scalar,
        "act_types":    act_types,
        "act_amounts":  act_amounts,
        "act_mask":     act_mask,
        "label_action": lbl_action,
        "label_amount": lbl_amount,
    }


def collate_fn(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack a list of tokenize() outputs into batched tensors (B, ...)."""
    keys = samples[0].keys()
    return {k: torch.stack([s[k] for s in samples]) for k in keys}


# ── Card string parser helper (used by dataset parsers) ───────────────────────

_CARD_RE = re.compile(r"\b([2-9TJQKA][cdhs])\b", re.IGNORECASE)


def parse_cards(text: str) -> List[int]:
    """Extract all card IDs from a string like 'Ks 7h 2d'."""
    return [card_to_id(m.group(1)) for m in _CARD_RE.finditer(text)]
