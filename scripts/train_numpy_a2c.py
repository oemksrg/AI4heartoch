"""不依赖 PyTorch 的最小 A2C（NumPy 线性策略）接管训练。

用法:
  python scripts/train_numpy_a2c.py --steps 20000 --n-games 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.numpy_policy import LinearPolicy  # noqa: E402


def evaluate(policy: LinearPolicy, episodes: int, seed: int, max_steps: int = 1500) -> dict:
    import taverngym

    env = taverngym.BattlegroundsEnv(seed=seed)
    rng = np.random.default_rng(seed)
    placements, rewards = [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        total = 0.0
        steps = 0
        while steps < max_steps:
            if steps > 0 and steps % 35 == 0 and obs["action_mask"][0]:
                action = 0  # END_TURN fallback to avoid softlocks
            else:
                action, _, _, _ = policy.act(
                    obs["features"], obs["action_mask"], rng, deterministic=True
                )
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            steps += 1
            if terminated or truncated:
                break
        placements.append(info.get("placement"))
        rewards.append(total)
    pl = [p for p in placements if p is not None]
    return {
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "mean_placement": float(np.mean(pl)) if pl else None,
        "placements": pl,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--n-games", type=int, default=16)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-episodes", type=int, default=15)
    p.add_argument("--out-dir", type=str, default="runs")
    args = p.parse_args()

    import taverngym

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = taverngym.BatchEnv(
        n_games=args.n_games,
        seed=args.seed,
        controlled=(0,),
        threads=args.threads,
    )
    policy = LinearPolicy(taverngym.OBS_FLOATS, taverngym.N_ACTIONS, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    n = env.n_slots
    buf = [[] for _ in range(n)]
    obs = env.observe()
    step = 0
    finished = 0
    updates = 0
    recent_r, recent_p = [], []
    t0 = time.perf_counter()

    print(f"numpy A2C | games={args.n_games} steps={args.steps}")
    while step < args.steps:
        ready = obs["ready"]
        actions = np.zeros(n, dtype=np.int64)
        meta = [None] * n
        for i in range(n):
            if not ready[i]:
                continue
            # 长轨迹强制结束回合，避免策略陷入 buy/move 循环
            force_end = len(buf[i]) > 0 and len(buf[i]) % 40 == 0 and bool(obs["action_mask"][i][0])
            a, logp, value, _ = policy.act(obs["features"][i], obs["action_mask"][i], rng)
            if force_end:
                a = 0
            actions[i] = a
            meta[i] = {
                "features": obs["features"][i].astype(np.float64),
                "mask": obs["action_mask"][i].copy(),
                "action": a,
                "logp": logp,
                "value": value,
            }

        rewards, dones, info = env.step(actions)
        step += int(ready.sum())
        obs = env.observe()

        for i in range(n):
            if meta[i] is None:
                continue
            item = meta[i]
            item["reward"] = float(rewards[i])
            buf[i].append(item)
            if not dones[i]:
                continue

            traj = buf[i]
            finished += 1
            rs = [t["reward"] for t in traj]
            recent_r.append(sum(rs))
            place = int(info["placement"][i])
            if place > 0:
                recent_p.append(place)

            G = 0.0
            returns = []
            for r in reversed(rs):
                G = r + args.gamma * G
                returns.append(G)
            returns.reverse()
            batch = []
            values = np.array([t["value"] for t in traj], dtype=np.float64)
            R = np.array(returns, dtype=np.float64)
            adv = R - values
            if len(adv) > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            for t, a_, ret in zip(traj, adv, R):
                batch.append(
                    {
                        "features": t["features"],
                        "mask": t["mask"],
                        "action": t["action"],
                        "adv": float(a_),
                        "ret": float(ret),
                    }
                )
            policy.update(batch, lr=args.lr)
            updates += 1
            buf[i] = []

            if updates % args.log_every == 0:
                mean_r = float(np.mean(recent_r[-50:])) if recent_r else 0.0
                mean_p = float(np.mean(recent_p[-50:])) if recent_p else float("nan")
                print(
                    f"upd={updates} step={step} finished={finished} "
                    f"meanR50={mean_r:+.3f} meanPlace50={mean_p:.2f} "
                    f"{time.perf_counter() - t0:.0f}s"
                )

    ckpt = out / "numpy_a2c_latest.npz"
    policy.save(str(ckpt))
    metrics = evaluate(policy, args.eval_episodes, args.seed + 10_000)
    (out / "numpy_a2c_eval.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=== eval ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
