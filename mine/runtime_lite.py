"""轻量 checkpoint 读取（仅推理需要的字段）。"""

from __future__ import annotations

import torch
import taverngym as tg


def engine_signature():
    return {
        "source_hash": tg.build_info()["source_hash"],
        "game_version": tg.game_version(),
        "features": tg.OBS_FLOATS,
        "cards": tg.OBS_CARDS,
        "actions": tg.N_ACTIONS,
        "vocabulary": tg.N_CARDS,
    }


def read_checkpoint(path, map_location="cpu"):
    try:
        saved = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        saved = torch.load(path, map_location=map_location)
    if saved.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format_version")
    if saved.get("engine") != engine_signature():
        raise ValueError(
            f"engine mismatch:\n  ckpt={saved.get('engine')}\n  local={engine_signature()}"
        )
    return saved
