"""评估交接包 PPO 权重（checkpoints/handoff_best.pt）。

用法（推荐 Anaconda 的可用 torch）:
  D:\\app\\anaconda\\python.exe scripts/eval_handoff_ppo.py --episodes 10
  D:\\app\\anaconda\\python.exe scripts/eval_handoff_ppo.py --compare --episodes 8
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))


def load_handoff_model(checkpoint: Path, device: str = "cpu"):
    from mine.config import Config
    from mine.model import ActorCritic, observation_tensors
    from mine.runtime_lite import read_checkpoint

    saved = read_checkpoint(checkpoint, map_location=device)
    config = Config.from_dict(saved["config"])
    model = ActorCritic(config.model).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    return model, saved, observation_tensors


def run_policy(env, choose_fn, episodes: int, seed: int, max_steps: int = 2000):
    placements, rewards, steps_all = [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        total = 0.0
        steps = 0
        while steps < max_steps:
            action = choose_fn(obs, steps)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            steps += 1
            if terminated or truncated:
                break
        placements.append(info.get("placement"))
        rewards.append(total)
        steps_all.append(steps)
    # placement==0 表示未正常结束（超时/守卫），不计入平均名次
    pl = [p for p in placements if p is not None and p > 0]
    unfinished = sum(1 for p in placements if not p)
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_placement": float(np.mean(pl)) if pl else None,
        "placements": placements,
        "finished_placements": pl,
        "unfinished": unfinished,
        "mean_steps": float(np.mean(steps_all)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT / "checkpoints" / "handoff_best.pt"),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--compare", action="store_true", help="同时对比 random / heuristic")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    import taverngym

    print(f"taverngym {taverngym.version()} source_hash={taverngym.build_info()['source_hash']}")
    print(f"torch {torch.__version__}")
    print(f"loading {args.checkpoint}")

    model, saved, observation_tensors = load_handoff_model(Path(args.checkpoint), args.device)
    print(
        f"global_step={saved['global_step']} "
        f"params={sum(p.numel() for p in model.parameters())} "
        f"model={saved['config']['model']}"
    )

    env = taverngym.BattlegroundsEnv(seed=args.seed)

    def handoff_action(obs, steps):
        batch = {
            "features": obs["features"][None].copy(),
            "cards": obs["cards"][None].copy(),
            "action_mask": obs["action_mask"][None].copy(),
        }
        with torch.inference_mode():
            dist, _ = model.policy(*observation_tensors(batch, torch.device(args.device)))
            if args.stochastic:
                action = int(dist.sample().item())
            else:
                action = int(dist.logits.argmax(dim=-1).item())
        if not obs["action_mask"][action]:
            action = int(np.flatnonzero(obs["action_mask"])[0])
        # 轻量防卡死
        if steps > 0 and steps % 45 == 0 and bool(obs["action_mask"][0]):
            return 0
        return action

    results = {
        "handoff_ppo": run_policy(env, handoff_action, args.episodes, args.seed),
    }
    print("handoff:", json.dumps(results["handoff_ppo"], ensure_ascii=False))

    if args.compare:
        rng = random.Random(args.seed)

        def random_action(obs, steps):
            return int(rng.choice(np.flatnonzero(obs["action_mask"])))

        baseline_path = ROOT / "scripts" / "takeover_baseline.py"
        spec = importlib.util.spec_from_file_location("takeover_baseline", baseline_path)
        baseline = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(baseline)

        def heuristic_action(obs, steps):
            if steps > 0 and steps % 40 == 0 and bool(obs["action_mask"][0]):
                return 0
            return baseline.heuristic_action(obs)

        results["random"] = run_policy(env, random_action, args.episodes, args.seed + 1000)
        results["heuristic"] = run_policy(env, heuristic_action, args.episodes, args.seed + 2000)
        print("random:", json.dumps(results["random"], ensure_ascii=False))
        print("heuristic:", json.dumps(results["heuristic"], ensure_ascii=False))

    out = ROOT / "runs" / "handoff_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "global_step": saved["global_step"],
        "episodes": args.episodes,
        "results": results,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== summary ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
