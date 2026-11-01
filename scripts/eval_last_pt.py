"""用 last.pt（Transformer 策略）在 BattlegroundsEnv 上评估。

依赖 Anaconda 的可用 torch；taverngym 来自本仓库 python/ + _core.pyd。

用法:
  D:\\app\\anaconda\\python.exe scripts/eval_last_pt.py --checkpoint last.pt --episodes 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "last.pt"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    import taverngym
    from train.tavern_transformer import load_policy

    print(f"taverngym {taverngym.version()} | torch {torch.__version__}")
    print(f"loading {args.checkpoint}")
    model = load_policy(args.checkpoint, device=args.device)
    meta = model._meta
    print(f"stage={meta.get('stage')} source_hash={meta.get('source_hash')}")
    print(f"local build source_hash={taverngym.build_info()['source_hash']}")

    env = taverngym.BattlegroundsEnv(seed=args.seed)
    placements, rewards, steps_all = [], [], []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        total = 0.0
        steps = 0
        while steps < args.max_steps:
            f = torch.as_tensor(obs["features"], device=args.device).unsqueeze(0)
            c = torch.as_tensor(obs["cards"], device=args.device, dtype=torch.long).unsqueeze(0)
            m = torch.as_tensor(obs["action_mask"], device=args.device).unsqueeze(0)
            actions, _, _ = model.act(f, c, m, deterministic=not args.stochastic)
            action = int(actions.item())
            if not obs["action_mask"][action]:
                legal = np.flatnonzero(obs["action_mask"])
                action = int(legal[0])
            # 打断 Freeze/Refresh/Cancel 死循环
            if action in (1, 2, 7) and steps > 0 and steps % 8 == 0 and bool(obs["action_mask"][0]):
                action = 0
            elif steps > 0 and steps % 40 == 0 and bool(obs["action_mask"][0]):
                action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            steps += 1
            if terminated or truncated:
                break
        placements.append(info.get("placement"))
        rewards.append(total)
        steps_all.append(steps)
        print(
            f"ep={ep+1}/{args.episodes} placement={info.get('placement')} "
            f"reward={total:+.3f} steps={steps}"
        )

    pl = [p for p in placements if p is not None]
    summary = {
        "episodes": args.episodes,
        "mean_reward": float(np.mean(rewards)),
        "mean_placement": float(np.mean(pl)) if pl else None,
        "placements": pl,
        "mean_steps": float(np.mean(steps_all)),
        "checkpoint": str(args.checkpoint),
    }
    out = ROOT / "runs" / "last_pt_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
