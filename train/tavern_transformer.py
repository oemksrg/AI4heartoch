"""加载 AutoTavern 风格 last.pt（Transformer 策略）并做推理。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _slice_layout(layout: dict):
    g = layout["global_f"]
    q = layout["n_quests"] * layout["quest_f"]
    p = layout["pending_kinds"]
    global_n = g + q + p  # 68
    board_n = layout["n_board"] * layout["minion_f"]
    hand_n = layout["n_hand"] * layout["item_f"]
    shop_n = layout["n_shop"] * layout["item_f"]
    choice_n = layout["n_choice"] * layout["item_f"]
    opp_pub_n = layout["n_opp"] * layout["opp_f"]
    opp_board_n = layout["n_opp"] * layout["n_board"] * layout["minion_f"]
    assert global_n + board_n + hand_n + shop_n + choice_n + opp_pub_n + opp_board_n

    # card index offsets (must match crates/taverngym/src/obs.rs)
    # hero, powers(2), deity, pending_src, quests(3*3), secrets(5), board, hand, shop, choice, trinkets(8), opp heroes, opp boards
    c = 0
    c_hero = c
    c += 1
    c += layout.get("n_powers", 2)
    c += 2  # deity + pending source
    c += 3 * layout["n_quests"]
    c += layout.get("n_secrets", 5)
    c_board = c
    c += layout["n_board"]
    c_hand = c
    c += layout["n_hand"]
    c_shop = c
    c += layout["n_shop"]
    c_choice = c
    c += layout["n_choice"]
    c += 8  # trinkets
    c_opp = c
    c += layout["n_opp"]
    c_opp_board = c
    c += layout["n_opp"] * layout["n_board"]

    return {
        "global_n": global_n,
        "f_board": (global_n, global_n + board_n),
        "f_hand": (global_n + board_n, global_n + board_n + hand_n),
        "f_shop": (
            global_n + board_n + hand_n,
            global_n + board_n + hand_n + shop_n,
        ),
        "f_choice": (
            global_n + board_n + hand_n + shop_n,
            global_n + board_n + hand_n + shop_n + choice_n,
        ),
        "f_opp_pub": (
            global_n + board_n + hand_n + shop_n + choice_n,
            global_n + board_n + hand_n + shop_n + choice_n + opp_pub_n,
        ),
        "f_opp_board": (
            global_n + board_n + hand_n + shop_n + choice_n + opp_pub_n,
            global_n + board_n + hand_n + shop_n + choice_n + opp_pub_n + opp_board_n,
        ),
        "c_hero": c_hero,
        "c_board": c_board,
        "c_hand": c_hand,
        "c_shop": c_shop,
        "c_choice": c_choice,
        "c_opp": c_opp,
        "c_opp_board": c_opp_board,
    }


class CardEncoder(nn.Module):
    def __init__(self, n_cards: int, d: int, static_dim: int, text_dim: int):
        super().__init__()
        self.id_emb = nn.Embedding(n_cards, d, padding_idx=0)
        self.static_proj = nn.Linear(static_dim, d, bias=False)
        self.text_proj = nn.Linear(text_dim, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, card_ids: torch.Tensor, static: torch.Tensor, text: torch.Tensor):
        # card_ids [...], static [..., S], text [..., T]
        h = self.id_emb(card_ids) + self.static_proj(static) + self.text_proj(text)
        return self.norm(h)


class TavernTransformerPolicy(nn.Module):
    def __init__(self, layout: dict, arch: dict):
        super().__init__()
        self.layout = layout
        self.slices = _slice_layout(layout)
        d = arch["d"]
        layers = arch["layers"]
        heads = arch["heads"]
        n_cards = layout["n_cards"]
        n_actions = layout["n_actions"]
        static_dim = layout.get("static_dim", 66)
        # checkpoint static_proj is 66; prefer weight size when loading
        text_dim = layout.get("text_dim", 64)

        self.card_enc = CardEncoder(n_cards, d, static_dim=66, text_dim=text_dim)
        self.type_emb = nn.Embedding(6, d)
        self.slot_emb = nn.Embedding(17, d)
        self.proj = nn.ModuleDict(
            {
                "global": nn.Linear(68, d),
                "board": nn.Linear(layout["minion_f"], d),
                "hand": nn.Linear(layout["item_f"], d),
                "shop": nn.Linear(layout["item_f"], d),
                "choice": nn.Linear(layout["item_f"], d),
                "opp": nn.Linear(71, d),
            }
        )
        self.in_norm = nn.LayerNorm(d)

        class Trunk(nn.Module):
            def __init__(self):
                super().__init__()
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=heads,
                    dim_feedforward=d * 4,
                    batch_first=True,
                    activation="relu",
                )
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
                self.norm = nn.LayerNorm(d)

            def forward(self, x):
                return self.norm(self.encoder(x))

        self.trunk = Trunk()
        self.act_emb = nn.Embedding(n_actions, d)
        self.pos_emb = nn.Embedding(9, d)
        self.w_a = nn.Linear(d, d, bias=False)
        self.w_b = nn.Linear(d, d, bias=False)
        self.act_mlp = nn.Sequential(nn.Linear(d, d))
        self.act_out = nn.Linear(d, 1)
        self.value_head = nn.Linear(d, 1)
        self.placement_head = nn.Linear(d, 8)

        self.register_buffer("card_static", torch.zeros(n_cards, 66), persistent=False)
        self.register_buffer("card_text", torch.zeros(n_cards, text_dim), persistent=False)
        self._action_pos = self._build_action_pos(layout)

    @staticmethod
    def _build_action_pos(layout: dict) -> torch.Tensor:
        n = layout["n_actions"]
        pos = torch.full((n,), 8, dtype=torch.long)  # 8 = no position
        a_play = layout["a_play"]
        a_magnetize = layout["a_magnetize"]
        a_move = layout["a_move"]
        n_pos = layout["n_pos"]
        n_board = layout["n_board"]
        # play hand i at warband position j
        for a in range(a_play, a_magnetize):
            pos[a] = (a - a_play) % n_pos
        # magnetize: use board target index
        for a in range(a_magnetize, a_move):
            pos[a] = (a - a_magnetize) % n_board
        # move: destination index
        for a in range(a_move, n):
            pos[a] = (a - a_move) % n_board
        return pos

    def set_card_tables(self, static_np: np.ndarray, text_np: np.ndarray | None = None):
        # static_np: [n_cards, 66]
        st = torch.as_tensor(static_np, dtype=torch.float32)
        if st.shape != self.card_static.shape:
            raise ValueError(f"static shape {st.shape} != {tuple(self.card_static.shape)}")
        self.card_static.copy_(st)
        if text_np is not None:
            tx = torch.as_tensor(text_np, dtype=torch.float32)
            self.card_text.copy_(tx)

    def encode_cards(self, card_ids: torch.Tensor) -> torch.Tensor:
        flat = card_ids.clamp(0, self.card_static.shape[0] - 1)
        static = self.card_static[flat]
        text = self.card_text[flat]
        return self.card_enc(flat, static, text)

    def _tokens(self, features: torch.Tensor, cards: torch.Tensor):
        # features [B, F], cards [B, C]
        B = features.shape[0]
        sl = self.slices
        L = self.layout
        device = features.device

        g = features[:, : sl["global_n"]]
        board = features[:, sl["f_board"][0] : sl["f_board"][1]].view(B, L["n_board"], L["minion_f"])
        hand = features[:, sl["f_hand"][0] : sl["f_hand"][1]].view(B, L["n_hand"], L["item_f"])
        shop = features[:, sl["f_shop"][0] : sl["f_shop"][1]].view(B, L["n_shop"], L["item_f"])
        choice = features[:, sl["f_choice"][0] : sl["f_choice"][1]].view(B, L["n_choice"], L["item_f"])
        opp_pub = features[:, sl["f_opp_pub"][0] : sl["f_opp_pub"][1]].view(B, L["n_opp"], L["opp_f"])
        opp_board = features[:, sl["f_opp_board"][0] : sl["f_opp_board"][1]].view(
            B, L["n_opp"], L["n_board"], L["minion_f"]
        )
        # opp token features: public + first two seen minions = 9+31+31=71
        opp = torch.cat([opp_pub, opp_board[:, :, 0, :], opp_board[:, :, 1, :]], dim=-1)

        c_hero = cards[:, sl["c_hero"]]
        c_board = cards[:, sl["c_board"] : sl["c_board"] + L["n_board"]]
        c_hand = cards[:, sl["c_hand"] : sl["c_hand"] + L["n_hand"]]
        c_shop = cards[:, sl["c_shop"] : sl["c_shop"] + L["n_shop"]]
        c_choice = cards[:, sl["c_choice"] : sl["c_choice"] + L["n_choice"]]
        c_opp = cards[:, sl["c_opp"] : sl["c_opp"] + L["n_opp"]]

        toks = []
        types = []
        slots = []

        def add(x, card_ids, type_id, n_slots):
            e = x + self.encode_cards(card_ids)
            toks.append(e)
            types.append(torch.full((B, n_slots), type_id, device=device, dtype=torch.long))
            slots.append(
                torch.arange(n_slots, device=device).view(1, -1).expand(B, -1).clamp(max=16)
            )

        add(self.proj["global"](g).unsqueeze(1), c_hero.unsqueeze(1), 0, 1)
        add(self.proj["board"](board), c_board, 1, L["n_board"])
        add(self.proj["hand"](hand), c_hand, 2, L["n_hand"])
        add(self.proj["shop"](shop), c_shop, 3, L["n_shop"])
        add(self.proj["choice"](choice), c_choice, 4, L["n_choice"])
        add(self.proj["opp"](opp), c_opp, 5, L["n_opp"])

        x = torch.cat(toks, dim=1)
        x = x + self.type_emb(torch.cat(types, dim=1)) + self.slot_emb(torch.cat(slots, dim=1))
        x = self.in_norm(x)
        x = self.trunk(x)
        return x

    def forward(self, features: torch.Tensor, cards: torch.Tensor, action_mask: torch.Tensor | None = None):
        x = self._tokens(features, cards)
        h = x[:, 0]
        qa = self.w_a(h)  # [B, d]
        pos = self._action_pos.to(features.device)
        kb = self.w_b(self.act_emb.weight + self.pos_emb(pos))  # [A, d]
        # reverse-engineered: relu(q*k) then act_mlp then act_out
        # (plain mul→mlp→relu tended to Freeze-loop; this finishes games more reliably)
        z = self.act_mlp(F.relu(qa.unsqueeze(1) * kb.unsqueeze(0)))
        logits = self.act_out(z).squeeze(-1)
        if action_mask is not None:
            neg = torch.finfo(logits.dtype).min
            logits = torch.where(action_mask.bool(), logits, torch.full_like(logits, neg))
        value = self.value_head(h).squeeze(-1)
        placement = self.placement_head(h)
        return logits, value, placement

    @torch.no_grad()
    def act(self, features, cards, action_mask, deterministic: bool = True):
        logits, value, placement = self.forward(features, cards, action_mask)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
        return actions, value, placement


def load_card_static_table() -> np.ndarray:
    import taverngym

    dim, raw = taverngym.card_static()
    arr = np.frombuffer(raw, dtype="<f4").reshape(-1, dim)  # [n_cards, 66]
    return np.array(arr, dtype=np.float32, copy=True)


def load_policy(checkpoint: str | Path, device: str = "cpu") -> TavernTransformerPolicy:
    ckpt = torch.load(str(checkpoint), map_location=device)
    meta = ckpt["meta"]
    model = TavernTransformerPolicy(meta["layout"], meta["arch"])
    # static_dim in meta may be 61; weights use 66 — already hardcoded in CardEncoder
    missing, unexpected = model.load_state_dict(ckpt["policy"], strict=False)
    missing = [m for m in missing if "card_static" not in m and "card_text" not in m]
    if unexpected or missing:
        raise RuntimeError(f"load mismatch unexpected={unexpected[:8]} missing={missing[:8]}")
    model.set_card_tables(load_card_static_table())
    model.to(device)
    model.eval()
    model._meta = meta
    return model
