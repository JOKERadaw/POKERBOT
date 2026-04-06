"""
poker_ui.py
===========
Pygame-based visual interface for the PokerTransformer.

Layout
------
  LEFT  (360px) : Card picker grid — click to select, place in active slot
  CENTER(540px) : Poker table with card slots + action history
  RIGHT (400px) : Game-state controls, action editor, PREDICT button, results

Workflow
--------
  1. Click a card slot on the table  → it glows gold (becomes the active target)
  2. Click any card in the picker    → card is placed in the active slot;
                                       active slot advances automatically
  3. Right-click a slot              → clears it
  4. Set Position / Street in the right panel
  5. Fill Pot, Hero Stack, Villain Stack
  6. Add past actions with the action editor
  7. Click  PREDICT  →  model outputs action + probability bars + bet size

Run
---
  python -X utf8 poker_ui.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pygame
import torch

from poker_game_state import (
    GameState, PokerAction, tokenize,
    id_to_card, card_to_id,
    FOLD, CHECK, CALL, BET, RAISE,
    POSITIONS, PREFLOP, FLOP, TURN, RIVER,
    OUTPUT_ACTION_NAMES,
)
from poker_model import build_poker_transformer


# ─────────────────────────────────────────────────────────────────────────────
# Window & panel geometry
# ─────────────────────────────────────────────────────────────────────────────
WIN_W, WIN_H = 1300, 850
FPS          = 60

PICKER_X, PICKER_W = 0,   360
TABLE_X,  TABLE_W  = 360, 540
CTRL_X,   CTRL_W   = 900, 400


# ─────────────────────────────────────────────────────────────────────────────
# Color palette
# ─────────────────────────────────────────────────────────────────────────────
BG          = (12, 28, 12)
PANEL_LINE  = (45, 85, 45)
FELT        = (28, 88, 28)
FELT_EDGE   = (68, 44, 14)

CARD_FACE   = (252, 250, 240)
CARD_USED   = (62, 72, 62)
CARD_HOVER  = (195, 225, 195)
SLOT_EMPTY  = (20, 58, 20)
SLOT_ACTIVE = (255, 215, 0)   # gold

RED_SUIT    = (200, 35, 35)
BLK_SUIT    = (18, 18, 18)

TEXT        = (235, 235, 215)
TEXT_DIM    = (130, 155, 130)
GOLD        = (255, 215, 0)
WHITE       = (255, 255, 255)
BLACK       = (0, 0, 0)

BTN_N       = (36, 90, 36)
BTN_H       = (52, 125, 52)
BTN_SEL     = (155, 115, 12)
BTN_PRED    = (18, 155, 18)
BTN_PRED_H  = (28, 195, 28)
BTN_ADD     = (22, 105, 105)
BTN_ADD_H   = (32, 135, 135)
BTN_DEL     = (120, 30, 30)
BTN_DEL_H   = (160, 45, 45)

ACT_COLORS = {
    FOLD:  (165, 42, 42),
    CHECK: (55, 120, 55),
    CALL:  (45, 85, 160),
    BET:   (155, 105, 25),
    RAISE: (130, 45, 155),
}

STREET_NAMES = ["Preflop", "Flop", "Turn", "River"]
POS_NAMES    = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
ACT_NAMES    = ["Fold", "Check", "Call", "Bet", "Raise"]

# Suit → label/color
SUIT_LABEL = ["c", "d", "h", "s"]
SUIT_COLOR = [BLK_SUIT, RED_SUIT, RED_SUIT, BLK_SUIT]
RANK_LABEL = list("23456789TJQKA")


# ─────────────────────────────────────────────────────────────────────────────
# Tiny widgets
# ─────────────────────────────────────────────────────────────────────────────

class Button:
    """Simple rectangular button with hover and selected states."""

    def __init__(
        self,
        rect: Tuple[int, int, int, int],
        label: str,
        color_normal=BTN_N,
        color_hover=BTN_H,
        color_sel=BTN_SEL,
        font_size: int = 13,
        selected: bool = False,
        radius: int = 5,
    ):
        self.rect    = pygame.Rect(rect)
        self.label   = label
        self.cn      = color_normal
        self.ch      = color_hover
        self.cs      = color_sel
        self.sel     = selected
        self.radius  = radius
        self._font   = None
        self._fs     = font_size

    def _get_font(self):
        if self._font is None:
            self._font = pygame.font.SysFont("segoeui", self._fs, bold=True)
        return self._font

    def is_hovered(self) -> bool:
        return self.rect.collidepoint(pygame.mouse.get_pos())

    def was_clicked(self, event: pygame.event.Event) -> bool:
        return (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        )

    def draw(self, surf: pygame.Surface) -> None:
        if self.sel:
            color = self.cs
        elif self.is_hovered():
            color = self.ch
        else:
            color = self.cn
        pygame.draw.rect(surf, color, self.rect, border_radius=self.radius)
        pygame.draw.rect(surf, PANEL_LINE, self.rect, 1, border_radius=self.radius)
        txt = self._get_font().render(self.label, True, WHITE)
        surf.blit(
            txt,
            (self.rect.centerx - txt.get_width() // 2,
             self.rect.centery - txt.get_height() // 2),
        )


class TextInput:
    """Single-line numeric (or text) input with cursor blink."""

    def __init__(
        self,
        rect: Tuple[int, int, int, int],
        label: str = "",
        initial: str = "0",
        numeric: bool = True,
    ):
        self.rect    = pygame.Rect(rect)
        self.label   = label
        self.text    = initial
        self.focused = False
        self.numeric = numeric
        self._blink  = 0.0
        self._font   = None
        self._lfont  = None

    def _fonts(self):
        if self._font is None:
            self._font  = pygame.font.SysFont("segoeui", 14)
            self._lfont = pygame.font.SysFont("segoeui", 12)
        return self._font, self._lfont

    def update(self, dt: float) -> None:
        self._blink = (self._blink + dt) % 1.0

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.focused = self.rect.collidepoint(event.pos)
            return self.focused
        if not self.focused:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key in (pygame.K_RETURN, pygame.K_TAB, pygame.K_ESCAPE):
                self.focused = False
            elif self.numeric:
                if event.unicode in "0123456789":
                    self.text += event.unicode
                elif event.unicode == "." and "." not in self.text:
                    self.text += event.unicode
            else:
                self.text += event.unicode
        return True

    def get_float(self, default: float = 0.0) -> float:
        try:
            return float(self.text) if self.text else default
        except ValueError:
            return default

    def draw(self, surf: pygame.Surface) -> None:
        fn, lf = self._fonts()
        if self.label:
            ls = lf.render(self.label, True, TEXT_DIM)
            surf.blit(ls, (self.rect.x, self.rect.y - ls.get_height() - 3))
        bg = (20, 50, 20) if not self.focused else (28, 65, 28)
        pygame.draw.rect(surf, bg, self.rect, border_radius=4)
        border = GOLD if self.focused else (55, 95, 55)
        pygame.draw.rect(surf, border, self.rect, 2, border_radius=4)
        cursor = "|" if self.focused and self._blink < 0.5 else " "
        ts = fn.render(self.text + cursor, True, WHITE)
        surf.blit(ts, (self.rect.x + 6, self.rect.centery - ts.get_height() // 2))


# ─────────────────────────────────────────────────────────────────────────────
# Card drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rank_suit(card_id: int) -> Tuple[int, int]:
    return card_id // 4, card_id % 4


def _draw_card(
    surf: pygame.Surface,
    rect: pygame.Rect,
    card_id: Optional[int],
    active: bool = False,
    used: bool = False,
    hovered: bool = False,
    font_sm=None,
    font_xs=None,
) -> None:
    """Draw a card face (or empty slot) into rect."""
    radius = 5

    if card_id is None:
        # Empty slot
        color  = (12, 40, 12) if not active else (30, 55, 10)
        border = GOLD if active else (50, 90, 50)
        pygame.draw.rect(surf, color, rect, border_radius=radius)
        pygame.draw.rect(surf, border, rect, 2 if not active else 3, border_radius=radius)
        if active:
            # Pulsing "?" indicator
            if font_sm:
                q = font_sm.render("?", True, GOLD)
                surf.blit(q, (rect.centerx - q.get_width() // 2,
                              rect.centery - q.get_height() // 2))
        return

    rank, suit = _rank_suit(card_id)
    suit_clr   = SUIT_COLOR[suit]
    bg         = CARD_USED if used else (CARD_HOVER if hovered else CARD_FACE)
    txt_clr    = (100, 100, 100) if used else suit_clr

    pygame.draw.rect(surf, bg, rect, border_radius=radius)
    border = GOLD if active else ((80, 80, 80) if used else (180, 180, 160))
    bw     = 3 if active else 1
    pygame.draw.rect(surf, border, rect, bw, border_radius=radius)

    if font_sm and font_xs:
        rl = font_sm.render(RANK_LABEL[rank], True, txt_clr)
        sl = font_xs.render(SUIT_LABEL[suit], True, txt_clr)
        surf.blit(rl, (rect.x + 3, rect.y + 2))
        surf.blit(sl, (rect.x + 3, rect.y + rl.get_height() + 1))
        # center suit
        big = font_sm.render(SUIT_LABEL[suit], True, txt_clr)
        surf.blit(big, (rect.centerx - big.get_width() // 2,
                        rect.centery - big.get_height() // 2 + 4))


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_model() -> Tuple[torch.nn.Module, str]:
    """Build model, load latest checkpoint if available."""
    model  = build_poker_transformer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    ckpt_dir = Path("poker_checkpoints")
    checkpoints = sorted(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
    if checkpoints:
        latest = checkpoints[-1]
        try:
            state = torch.load(latest, map_location=device)
            model.load_state_dict(state["model_state_dict"])
            status = f"Loaded: {latest.name}"
        except Exception as e:
            status = f"Load failed: {e}"
    else:
        status = "No checkpoint — random weights"

    return model, device, status


# ─────────────────────────────────────────────────────────────────────────────
# Main UI class
# ─────────────────────────────────────────────────────────────────────────────

class PokerUI:

    # Card slot order: hole0, hole1, comm0..comm4
    N_SLOTS = 7

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("PokerTransformer — Interactive Inference")
        self.clock  = pygame.time.Clock()

        # Fonts (initialised here so pygame is ready)
        self.f_title  = pygame.font.SysFont("segoeui", 16, bold=True)
        self.f_normal = pygame.font.SysFont("segoeui", 13)
        self.f_small  = pygame.font.SysFont("segoeui", 12)
        self.f_xs     = pygame.font.SysFont("segoeui", 10)
        self.f_card   = pygame.font.SysFont("segoeui", 11, bold=True)
        self.f_cardxs = pygame.font.SysFont("segoeui", 9)
        self.f_result = pygame.font.SysFont("segoeui", 22, bold=True)
        self.f_bar    = pygame.font.SysFont("segoeui", 12)

        # ── Game state ───────────────────────────────────────────────────────
        self.slots: List[Optional[int]] = [None] * self.N_SLOTS
        self.active_slot: Optional[int] = 0          # which slot awaits a card
        self.position_idx : int = 1                  # HJ default
        self.street_idx   : int = RIVER
        self.act_pos_idx  : int = 0                  # action editor: actor
        self.act_type_idx : int = 2                  # action editor: type (call)
        self.actions: List[Tuple[str, int, float]] = []   # (pos_name, type, amt)

        # ── Text inputs ──────────────────────────────────────────────────────
        ix = CTRL_X + 10
        self.inp_pot  = TextInput((ix, 260, 175, 28), "Pot (chips)",   "24")
        self.inp_hero = TextInput((ix + 190, 260, 175, 28), "Hero Stack",  "88")
        self.inp_vill = TextInput((ix, 320, 175, 28), "Villain Stack", "88")
        self.inp_stk  = TextInput((ix + 190, 320, 175, 28), "Starting Stack", "100")
        self.inp_amt  = TextInput((ix, 535, 360, 28), "Amount (chips — for call/bet/raise)", "0")
        self.all_inputs = [self.inp_pot, self.inp_hero, self.inp_vill, self.inp_stk, self.inp_amt]

        # ── Right-panel buttons ──────────────────────────────────────────────
        self._build_ctrl_buttons()

        # ── Card picker hover tracking ────────────────────────────────────────
        self.hover_card: Optional[int] = None   # card_id under mouse in picker

        # ── Prediction result ─────────────────────────────────────────────────
        self.result: Optional[dict] = None
        self.error : str            = ""

        # ── Model ─────────────────────────────────────────────────────────────
        print("[PokerUI] Loading model …")
        self.model, self.device, self.model_status = _load_model()
        print(f"[PokerUI] {self.model_status}")

        # ── Action history scroll ─────────────────────────────────────────────
        self.act_scroll = 0   # first visible action index

    # ── Button construction ────────────────────────────────────────────────────

    def _build_ctrl_buttons(self) -> None:
        ix = CTRL_X + 10
        bw = 58   # position button width
        bg = 4    # gap

        # Hero position (6 buttons, 1 row)
        self.pos_btns = [
            Button((ix + i * (bw + bg), 62, bw, 26), POS_NAMES[i],
                   selected=(i == self.position_idx))
            for i in range(6)
        ]

        # Street (4 buttons, 1 row), slightly wider
        sw = 88
        self.street_btns = [
            Button((ix + i * (sw + 4), 120, sw, 26), STREET_NAMES[i],
                   selected=(i == self.street_idx))
            for i in range(4)
        ]

        # Action editor — actor position (6 small buttons, 2 rows)
        sbw = 57
        self.act_pos_btns = [
            Button((ix + (i % 3) * (sbw + 4), 400 + (i // 3) * 32, sbw, 26),
                   POS_NAMES[i], font_size=12, selected=(i == self.act_pos_idx))
            for i in range(6)
        ]

        # Action editor — action type (5 buttons, 1 row)
        atw = 70
        act_type_colors = [
            (BTN_N, BTN_H),    # fold
            (BTN_N, BTN_H),    # check
            (BTN_N, BTN_H),    # call
            (BTN_N, BTN_H),    # bet
            (BTN_N, BTN_H),    # raise
        ]
        self.act_type_btns = [
            Button((ix + i * (atw + 3), 467, atw, 26), ACT_NAMES[i],
                   font_size=12,
                   color_normal=ACT_COLORS[i],
                   color_hover=tuple(min(c + 35, 255) for c in ACT_COLORS[i]),
                   selected=(i == self.act_type_idx))
            for i in range(5)
        ]

        # ADD ACTION button
        self.btn_add = Button(
            (ix, 575, 175, 32), "ADD ACTION",
            color_normal=BTN_ADD, color_hover=BTN_ADD_H, font_size=13
        )
        # CLEAR ALL actions button
        self.btn_clear = Button(
            (ix + 185, 575, 175, 32), "CLEAR ALL",
            color_normal=BTN_DEL, color_hover=BTN_DEL_H, font_size=13
        )

        # PREDICT button
        self.btn_predict = Button(
            (ix, 625, 360, 42), "PREDICT",
            color_normal=BTN_PRED, color_hover=BTN_PRED_H, font_size=18
        )

    # ── Card picker geometry ───────────────────────────────────────────────────

    CARD_W, CARD_H = 25, 36
    CARD_GAP_X     = 2
    CARD_GAP_Y     = 4
    GRID_X         = 12
    GRID_Y         = 90

    def _card_rect(self, rank: int, suit: int) -> pygame.Rect:
        x = self.GRID_X + rank * (self.CARD_W + self.CARD_GAP_X)
        y = self.GRID_Y + suit * (self.CARD_H + self.CARD_GAP_Y)
        return pygame.Rect(x, y, self.CARD_W, self.CARD_H)

    def _card_at_pos(self, pos: Tuple[int, int]) -> Optional[int]:
        """Return card_id at mouse position in the picker, or None."""
        if not (0 <= pos[0] < PICKER_W and 0 <= pos[1] < WIN_H):
            return None
        for suit in range(4):
            for rank in range(13):
                if self._card_rect(rank, suit).collidepoint(pos):
                    return rank * 4 + suit
        return None

    # ── Table card slot geometry ───────────────────────────────────────────────

    SLOT_W, SLOT_H = 52, 72
    SLOT_GAP       = 8

    def _slot_rect(self, slot_idx: int) -> pygame.Rect:
        """Pixel rect for a card slot on the poker table."""
        if slot_idx < 2:
            # Hole cards — centered at bottom of table area
            total = 2 * self.SLOT_W + self.SLOT_GAP
            x0    = TABLE_X + (TABLE_W - total) // 2
            return pygame.Rect(
                x0 + slot_idx * (self.SLOT_W + self.SLOT_GAP),
                345, self.SLOT_W, self.SLOT_H
            )
        else:
            # Community cards (slots 2-6)
            ci    = slot_idx - 2
            total = 5 * self.SLOT_W + 4 * self.SLOT_GAP
            x0    = TABLE_X + (TABLE_W - total) // 2
            return pygame.Rect(
                x0 + ci * (self.SLOT_W + self.SLOT_GAP),
                205, self.SLOT_W, self.SLOT_H
            )

    # ── Cards currently in use ─────────────────────────────────────────────────

    def _used_cards(self) -> set:
        return {c for c in self.slots if c is not None}

    # ── Event handling ─────────────────────────────────────────────────────────

    def _handle_event(self, event: pygame.event.Event) -> bool:
        """Returns False to quit."""
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return False

        # Scroll action history with mouse wheel in center panel
        if event.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            if TABLE_X <= mx < TABLE_X + TABLE_W and my > 440:
                self.act_scroll = max(0, self.act_scroll - event.y)

        # Pass keyboard events to focused text input first
        for inp in self.all_inputs:
            if inp.focused:
                inp.handle_event(event)
                return True

        # Text input focus (click)
        for inp in self.all_inputs:
            if event.type == pygame.MOUSEBUTTONDOWN:
                inp.handle_event(event)

        # ── Card picker click → place card in active slot ──────────────────
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            card = self._card_at_pos(event.pos)
            if card is not None and self.active_slot is not None:
                if card not in self._used_cards():
                    self.slots[self.active_slot] = card
                    # Advance to next empty slot
                    next_slot = None
                    for s in range(self.active_slot + 1, self.N_SLOTS):
                        if self.slots[s] is None:
                            next_slot = s
                            break
                    self.active_slot = next_slot
                return True

        # ── Table slot click → make active ────────────────────────────────
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for s in range(self.N_SLOTS):
                if self._slot_rect(s).collidepoint(event.pos):
                    self.active_slot = s
                    return True

        # ── Right-click table slot → clear ────────────────────────────────
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            for s in range(self.N_SLOTS):
                if self._slot_rect(s).collidepoint(event.pos):
                    self.slots[s] = None
                    if self.active_slot is None:
                        self.active_slot = s
                    return True

        # ── Position buttons ───────────────────────────────────────────────
        for i, btn in enumerate(self.pos_btns):
            if btn.was_clicked(event):
                self.position_idx = i
                for b in self.pos_btns:
                    b.sel = False
                btn.sel = True

        # ── Street buttons ─────────────────────────────────────────────────
        for i, btn in enumerate(self.street_btns):
            if btn.was_clicked(event):
                self.street_idx = i
                for b in self.street_btns:
                    b.sel = False
                btn.sel = True

        # ── Action editor: actor position ──────────────────────────────────
        for i, btn in enumerate(self.act_pos_btns):
            if btn.was_clicked(event):
                self.act_pos_idx = i
                for b in self.act_pos_btns:
                    b.sel = False
                btn.sel = True

        # ── Action editor: action type ─────────────────────────────────────
        for i, btn in enumerate(self.act_type_btns):
            if btn.was_clicked(event):
                self.act_type_idx = i
                for b in self.act_type_btns:
                    b.sel = False
                btn.sel = True

        # ── ADD ACTION ─────────────────────────────────────────────────────
        if self.btn_add.was_clicked(event):
            amt = self.inp_amt.get_float(0.0)
            self.actions.append((POS_NAMES[self.act_pos_idx], self.act_type_idx, amt))
            self.act_scroll = max(0, len(self.actions) - 8)
            self.error = ""

        # ── CLEAR ALL ──────────────────────────────────────────────────────
        if self.btn_clear.was_clicked(event):
            self.actions.clear()
            self.act_scroll = 0

        # ── Right-click action entry → delete ─────────────────────────────
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            mx, my = event.pos
            if TABLE_X <= mx < TABLE_X + TABLE_W and 455 <= my < 455 + 8 * 34:
                idx = self.act_scroll + (my - 455) // 34
                if 0 <= idx < len(self.actions):
                    self.actions.pop(idx)
                    self.act_scroll = max(0, min(self.act_scroll, len(self.actions) - 1))

        # ── PREDICT ────────────────────────────────────────────────────────
        if self.btn_predict.was_clicked(event):
            self._predict()

        return True

    # ── Inference ──────────────────────────────────────────────────────────────

    def _predict(self) -> None:
        self.result = None
        self.error  = ""

        # Validate hole cards
        if self.slots[0] is None or self.slots[1] is None:
            self.error = "Need both hole cards!"
            return

        hole_cards = [self.slots[0], self.slots[1]]
        community  = [self.slots[i] for i in range(2, 7) if self.slots[i] is not None]

        # Community count must be 0, 3, 4, or 5
        if len(community) == 1 or len(community) == 2:
            self.error = "Community: place 0, 3, 4, or 5 cards."
            return

        poker_actions = [
            PokerAction(action_type=t, amount=a)
            for (_, t, a) in self.actions
        ]

        pot  = self.inp_pot.get_float(10.0)
        hero = self.inp_hero.get_float(90.0)
        vill = self.inp_vill.get_float(90.0)
        stk  = self.inp_stk.get_float(100.0)

        gs = GameState(
            hole_cards      = hole_cards,
            community_cards = community,
            actions         = poker_actions,
            position        = self.position_idx,
            pot             = pot,
            hero_stack      = hero,
            villain_stack   = vill,
            street          = self.street_idx,
            starting_stack  = stk,
        )

        sample = tokenize(gs)

        with torch.no_grad():
            card_ids    = sample["card_ids"].unsqueeze(0).to(self.device)
            ctx_disc    = sample["ctx_disc"].unsqueeze(0).to(self.device)
            ctx_scalar  = sample["ctx_scalar"].unsqueeze(0).to(self.device)
            act_types   = sample["act_types"].unsqueeze(0).to(self.device)
            act_amounts = sample["act_amounts"].unsqueeze(0).to(self.device)
            act_mask    = sample["act_mask"].unsqueeze(0).to(self.device)

            logits, amount = self.model(
                card_ids, ctx_disc, ctx_scalar, act_types, act_amounts, act_mask
            )

        probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
        act_i  = int(torch.argmax(logits).item())
        amt_f  = amount.item()

        self.result = {
            "action_idx":   act_i,
            "action_name":  OUTPUT_ACTION_NAMES[act_i],
            "probs":        probs,
            "bet_fraction": amt_f,
            "bet_chips":    amt_f * pot,
        }

    # ── Drawing helpers ────────────────────────────────────────────────────────

    def _draw_left_panel(self) -> None:
        surf = self.screen
        used = self._used_cards()

        # Title
        t = self.f_title.render("CARD PICKER", True, GOLD)
        surf.blit(t, (PICKER_X + 10, 10))

        # Active slot label
        if self.active_slot is not None:
            labels = ["Hole 1", "Hole 2", "Flop 1", "Flop 2", "Flop 3", "Turn", "River"]
            label  = f"Placing: {labels[self.active_slot]}"
            ls = self.f_small.render(label, True, GOLD)
        else:
            ls = self.f_small.render("All cards placed  (click a slot to replace)", True, TEXT_DIM)
        surf.blit(ls, (PICKER_X + 10, 34))

        # Column headers (rank labels)
        for rank in range(13):
            rx = self.GRID_X + rank * (self.CARD_W + self.CARD_GAP_X) + self.CARD_W // 2
            rl = self.f_xs.render(RANK_LABEL[rank], True, TEXT_DIM)
            surf.blit(rl, (rx - rl.get_width() // 2, self.GRID_Y - 13))

        # Row headers (suit labels)
        suit_clr_labels = [BLK_SUIT, RED_SUIT, RED_SUIT, BLK_SUIT]
        suit_row_labels = ["c", "d", "h", "s"]
        for suit in range(4):
            ry = self.GRID_Y + suit * (self.CARD_H + self.CARD_GAP_Y) + self.CARD_H // 2
            sl = self.f_small.render(suit_row_labels[suit], True, suit_clr_labels[suit]
                                     if suit_clr_labels[suit] != BLK_SUIT else TEXT_DIM)
            surf.blit(sl, (self.GRID_X - 14, ry - sl.get_height() // 2))

        # Card grid
        mx, my = pygame.mouse.get_pos()
        for suit in range(4):
            for rank in range(13):
                cid    = rank * 4 + suit
                cr     = self._card_rect(rank, suit)
                is_used    = cid in used
                is_hovered = cr.collidepoint(mx, my) and not is_used and 0 <= mx < PICKER_W
                is_active  = (cid == self.active_slot)   # never true here, but kept for consistency
                _draw_card(
                    surf, cr, cid,
                    active=False,
                    used=is_used,
                    hovered=is_hovered,
                    font_sm=self.f_card,
                    font_xs=self.f_cardxs,
                )

        # Instructions
        lines = [
            "Click a table slot → gold = active",
            "Click a card here  → place in slot",
            "Right-click slot   → clear card",
        ]
        y0 = self.GRID_Y + 4 * (self.CARD_H + self.CARD_GAP_Y) + 18
        for i, line in enumerate(lines):
            ls = self.f_xs.render(line, True, TEXT_DIM)
            surf.blit(ls, (PICKER_X + 10, y0 + i * 16))

    def _draw_center_panel(self) -> None:
        surf = self.screen

        # Panel title
        t = self.f_title.render("POKER TABLE", True, GOLD)
        surf.blit(t, (TABLE_X + 10, 10))

        # ── Table oval ──────────────────────────────────────────────────────
        table_rect = pygame.Rect(TABLE_X + 22, 38, TABLE_W - 44, 305)
        # Outer edge (wood)
        edge_rect = table_rect.inflate(12, 12)
        pygame.draw.ellipse(surf, FELT_EDGE, edge_rect)
        # Felt
        pygame.draw.ellipse(surf, FELT, table_rect)
        # Inner highlight
        pygame.draw.ellipse(surf, tuple(min(c + 10, 255) for c in FELT),
                            table_rect.inflate(-20, -20), 2)

        # Community card labels
        cl = self.f_xs.render("COMMUNITY", True, TEXT_DIM)
        surf.blit(cl, (TABLE_X + TABLE_W // 2 - cl.get_width() // 2, 185))

        # Community card slots (slots 2-6)
        for s in range(2, 7):
            sr = self._slot_rect(s)
            is_active = (s == self.active_slot)
            _draw_card(
                surf, sr, self.slots[s],
                active=is_active,
                font_sm=self.f_card, font_xs=self.f_cardxs,
            )
            if is_active:
                pygame.draw.rect(surf, GOLD, sr, 3, border_radius=5)

        # Pot display
        pot_val  = self.inp_pot.get_float(0)
        pot_surf = self.f_normal.render(f"Pot: {pot_val:.1f}", True, GOLD)
        surf.blit(pot_surf, (TABLE_X + TABLE_W // 2 - pot_surf.get_width() // 2, 298))

        # Hero label + slots
        hl = self.f_xs.render("HERO", True, TEXT_DIM)
        surf.blit(hl, (TABLE_X + TABLE_W // 2 - hl.get_width() // 2, 333))
        for s in range(2):
            sr = self._slot_rect(s)
            is_active = (s == self.active_slot)
            _draw_card(
                surf, sr, self.slots[s],
                active=is_active,
                font_sm=self.f_card, font_xs=self.f_cardxs,
            )
            if is_active:
                pygame.draw.rect(surf, GOLD, sr, 3, border_radius=5)

        # ── Action history ─────────────────────────────────────────────────
        pygame.draw.line(surf, PANEL_LINE, (TABLE_X, 432), (TABLE_X + TABLE_W, 432))
        ah = self.f_title.render("ACTION HISTORY", True, TEXT)
        surf.blit(ah, (TABLE_X + 10, 438))
        hint = self.f_xs.render("right-click to delete  |  scroll with wheel", True, TEXT_DIM)
        surf.blit(hint, (TABLE_X + TABLE_W - hint.get_width() - 8, 442))

        max_visible = 8
        y0          = 458
        row_h       = 33
        visible     = self.actions[self.act_scroll: self.act_scroll + max_visible]

        if not self.actions:
            empty = self.f_small.render("No actions yet — add them in the right panel.", True, TEXT_DIM)
            surf.blit(empty, (TABLE_X + 10, y0 + 4))
        else:
            for i, (pos_name, act_type, amt) in enumerate(visible):
                row_y = y0 + i * row_h
                row_r = pygame.Rect(TABLE_X + 5, row_y, TABLE_W - 10, row_h - 3)
                bg    = (20, 55, 20) if i % 2 == 0 else (16, 45, 16)
                pygame.draw.rect(surf, bg, row_r, border_radius=4)

                # Position badge
                ac = ACT_COLORS[act_type]
                badge = pygame.Rect(TABLE_X + 8, row_y + 4, 38, 22)
                pygame.draw.rect(surf, (30, 70, 30), badge, border_radius=3)
                pygame.draw.rect(surf, ac, badge, 1, border_radius=3)
                pn = self.f_small.render(pos_name, True, ac)
                surf.blit(pn, (badge.centerx - pn.get_width() // 2,
                               badge.centery - pn.get_height() // 2))

                # Action label
                act_lbl = f"{ACT_NAMES[act_type]}"
                if amt > 0:
                    act_lbl += f"  {amt:.1f}"
                al = self.f_normal.render(act_lbl, True, ac)
                surf.blit(al, (TABLE_X + 55, row_y + row_h // 2 - al.get_height() // 2))

                # Global index
                gi = self.f_xs.render(f"#{self.act_scroll + i + 1}", True, TEXT_DIM)
                surf.blit(gi, (TABLE_X + TABLE_W - 18 - gi.get_width(), row_y + row_h // 2 - gi.get_height() // 2))

            # Scroll indicator
            if len(self.actions) > max_visible:
                total_r = pygame.Rect(TABLE_X + TABLE_W - 8, y0, 5, max_visible * row_h)
                pygame.draw.rect(surf, (40, 70, 40), total_r, border_radius=2)
                frac   = self.act_scroll / max(1, len(self.actions) - max_visible)
                th     = max(20, int(total_r.height * max_visible / len(self.actions)))
                ty     = total_r.y + int(frac * (total_r.height - th))
                thumb  = pygame.Rect(total_r.x, ty, 5, th)
                pygame.draw.rect(surf, GOLD, thumb, border_radius=2)

    def _draw_right_panel(self) -> None:
        surf = self.screen
        ix   = CTRL_X + 10

        # Title
        t = self.f_title.render("GAME STATE", True, GOLD)
        surf.blit(t, (CTRL_X + 10, 10))

        # ── Position ────────────────────────────────────────────────────────
        pl = self.f_small.render("Position (Hero seat):", True, TEXT_DIM)
        surf.blit(pl, (ix, 44))
        for btn in self.pos_btns:
            btn.draw(surf)

        # ── Street ──────────────────────────────────────────────────────────
        sl = self.f_small.render("Street:", True, TEXT_DIM)
        surf.blit(sl, (ix, 102))
        for btn in self.street_btns:
            btn.draw(surf)

        # ── Scalar inputs ────────────────────────────────────────────────────
        for inp in [self.inp_pot, self.inp_hero, self.inp_vill, self.inp_stk]:
            inp.draw(surf)

        # ── Separator ───────────────────────────────────────────────────────
        pygame.draw.line(surf, PANEL_LINE, (CTRL_X + 5, 364), (CTRL_X + CTRL_W - 5, 364))
        ah = self.f_title.render("ADD ACTION", True, TEXT)
        surf.blit(ah, (ix, 370))

        # Actor label
        al = self.f_small.render("Actor:", True, TEXT_DIM)
        surf.blit(al, (ix, 385))
        for btn in self.act_pos_btns:
            btn.draw(surf)

        # Action type label
        tl = self.f_small.render("Type:", True, TEXT_DIM)
        surf.blit(tl, (ix, 453))
        for btn in self.act_type_btns:
            btn.draw(surf)

        # Amount input
        self.inp_amt.draw(surf)

        # ADD + CLEAR buttons
        self.btn_add.draw(surf)
        self.btn_clear.draw(surf)

        # ── Separator ───────────────────────────────────────────────────────
        pygame.draw.line(surf, PANEL_LINE, (CTRL_X + 5, 614), (CTRL_X + CTRL_W - 5, 614))

        # ── PREDICT button ───────────────────────────────────────────────────
        self.btn_predict.draw(surf)

        # Model status
        ms = self.f_xs.render(self.model_status, True, TEXT_DIM)
        surf.blit(ms, (ix, 674))

        # Error message
        if self.error:
            es = self.f_small.render(self.error, True, (220, 70, 70))
            surf.blit(es, (ix, 692))

        # ── Prediction result ────────────────────────────────────────────────
        if self.result:
            r  = self.result
            ry = 710

            # Predicted action (large)
            act_name = r["action_name"].upper()
            ac       = ACT_COLORS.get(r["action_idx"], (100, 200, 100))
            act_surf = self.f_result.render(f"=> {act_name}", True, ac)
            surf.blit(act_surf, (ix, ry))
            ry += act_surf.get_height() + 6

            # Probability bars
            bar_w  = CTRL_W - 30
            bar_h  = 15
            labels = ["fold", "chk/call", "bet", "raise"]
            for i, (lbl, prob) in enumerate(zip(labels, r["probs"])):
                by = ry + i * (bar_h + 5)
                # Background
                pygame.draw.rect(surf, (25, 55, 25), (ix, by, bar_w, bar_h), border_radius=3)
                # Filled bar
                fw  = int(bar_w * prob)
                bc  = ACT_COLORS.get(i, (80, 150, 80))
                if fw > 0:
                    pygame.draw.rect(surf, bc, (ix, by, fw, bar_h), border_radius=3)
                # Highlight selected
                brd = GOLD if i == r["action_idx"] else (50, 90, 50)
                pygame.draw.rect(surf, brd, (ix, by, bar_w, bar_h), 1 if i != r["action_idx"] else 2, border_radius=3)
                # Label + percent
                lt = self.f_bar.render(f"{lbl}", True, WHITE)
                pt = self.f_bar.render(f"{prob*100:.0f}%", True, WHITE)
                surf.blit(lt, (ix + 4, by + bar_h // 2 - lt.get_height() // 2))
                surf.blit(pt, (ix + bar_w - pt.get_width() - 4, by + bar_h // 2 - pt.get_height() // 2))

            # Bet size (if bet or raise)
            if r["action_idx"] in (2, 3):
                ry2 = ry + 4 * (bar_h + 5) + 6
                bs  = self.f_small.render(
                    f"Bet: {r['bet_fraction']:.2f}x pot = {r['bet_chips']:.1f} chips",
                    True, GOLD
                )
                surf.blit(bs, (ix, ry2))

    def _draw_panel_lines(self) -> None:
        """Vertical separators between panels."""
        pygame.draw.line(self.screen, PANEL_LINE, (PICKER_W, 0), (PICKER_W, WIN_H), 1)
        pygame.draw.line(self.screen, PANEL_LINE, (TABLE_X + TABLE_W, 0), (TABLE_X + TABLE_W, WIN_H), 1)

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        running  = True
        prev_t   = pygame.time.get_ticks()

        while running:
            now    = pygame.time.get_ticks()
            dt     = (now - prev_t) / 1000.0
            prev_t = now

            for event in pygame.event.get():
                if not self._handle_event(event):
                    running = False

            # Update cursor blink
            for inp in self.all_inputs:
                inp.update(dt)

            # Draw
            self.screen.fill(BG)
            self._draw_left_panel()
            self._draw_center_panel()
            self._draw_right_panel()
            self._draw_panel_lines()

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ui = PokerUI()
    ui.run()
