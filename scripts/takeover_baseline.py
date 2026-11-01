"""BattlegroundsEnv 接管起步：用简单规则策略控制 1 席，其余 7 席为脚本 bot。

用法（先激活 .venv）:
  .venv\\Scripts\\python.exe scripts\\takeover_baseline.py
  .venv\\Scripts\\python.exe scripts\\takeover_baseline.py --episodes 20 --seed 0
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import taverngym


# 动作区间（与 docs/TAVERNGYM.md 一致）
END_TURN = 0
REFRESH = 1
FREEZE = 2
UPGRADE = 3
HERO_POWER = range(4, 6)
BUY = range(8, 16)
SELL = range(16, 23)
ACTIVATE = range(23, 30)
CHOOSE = range(30, 46)
PLAY = range(46, 126)
MAGNETIZE = range(126, 196)
MOVE = range(196, 245)


def first_legal(mask: np.ndarray, lo: int, hi: int) -> int | None:
    for a in range(lo, hi):
        if mask[a]:
            return a
    return None


def heuristic_action(obs: dict) -> int:
    """优先处理强制选择，再买牌/升级/出牌，最后结束回合。"""
    mask = obs["action_mask"]
    if not mask.any():
        raise RuntimeError("no legal actions")

    # 有待选选项时必须先选
    a = first_legal(mask, CHOOSE.start, CHOOSE.stop)
    if a is not None:
        return a

    # 能出牌 / 磁力贴合优先
    a = first_legal(mask, PLAY.start, PLAY.stop)
    if a is not None:
        return a
    a = first_legal(mask, MAGNETIZE.start, MAGNETIZE.stop)
    if a is not None:
        return a

    # 买酒馆、升级、英雄技能、激活
    a = first_legal(mask, BUY.start, BUY.stop)
    if a is not None:
        return a
    if mask[UPGRADE]:
        return UPGRADE
    a = first_legal(mask, HERO_POWER.start, HERO_POWER.stop)
    if a is not None:
        return a
    a = first_legal(mask, ACTIVATE.start, ACTIVATE.stop)
    if a is not None:
        return a

    # 刷新 / 冻结（比卖随从更稳妥）
    if mask[REFRESH]:
        return REFRESH
    if mask[FREEZE]:
        return FREEZE

    # 移动站位，再卖（缺钱时）
    a = first_legal(mask, MOVE.start, MOVE.stop)
    if a is not None:
        return a
    a = first_legal(mask, SELL.start, SELL.stop)
    if a is not None:
        return a

    if mask[END_TURN]:
        return END_TURN
    return int(mask.nonzero()[0][0])


def play_episode(
    env: taverngym.BattlegroundsEnv,
    seed: int | None = None,
    verbose: bool = False,
    max_steps: int = 2000,
):
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    while steps < max_steps:
        # 长时间不结束回合时强制 end turn，避免规则策略卡死
        if steps > 0 and steps % 40 == 0 and obs["action_mask"][END_TURN]:
            action = END_TURN
        else:
            action = heuristic_action(obs)
        if verbose and steps < 5:
            print(f"  step={steps} turn={info.get('turn')} action={action} ({taverngym.action_name(action)})")
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break
    return {
        "placement": info.get("placement"),
        "reward": total_reward,
        "steps": steps,
        "turn": info.get("turn"),
    }


def main():
    parser = argparse.ArgumentParser(description="Rule-based takeover baseline on BattlegroundsEnv")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent", type=int, default=0, help="controlled seat index")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"taverngym {taverngym.version()} | game {taverngym.game_version()}")
    print(f"obs sizes: {taverngym.BattlegroundsEnv().observation_sizes}")
    print(f"running {args.episodes} episode(s), seed={args.seed}, agent seat={args.agent}")

    env = taverngym.BattlegroundsEnv(seed=args.seed, agent=args.agent)
    results = []
    for i in range(args.episodes):
        ep_seed = args.seed + i
        result = play_episode(env, seed=ep_seed, verbose=args.verbose and i == 0)
        results.append(result)
        print(
            f"episode {i + 1}/{args.episodes}: placement={result['placement']} "
            f"reward={result['reward']:+.3f} steps={result['steps']} turn={result['turn']}"
        )

    placements = [r["placement"] for r in results if r["placement"] is not None]
    rewards = [r["reward"] for r in results]
    print("--- summary ---")
    print(f"mean reward: {np.mean(rewards):+.3f}")
    if placements:
        print(f"mean placement: {np.mean(placements):.2f} (1=best, 8=worst)")
        print(f"placement counts: {dict(sorted(Counter(placements).items()))}")


if __name__ == "__main__":
    main()
