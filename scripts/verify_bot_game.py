"""跑一局官方脚本 bot 对局，验证 taverngym 安装是否可用。"""

from __future__ import annotations

import time


def main() -> None:
    import taverngym

    print(f"taverngym version: {taverngym.version()}")
    print(f"game version: {taverngym.game_version()}")
    print(f"server rules: {taverngym.server_rules_version()}")
    print(f"N_ACTIONS={taverngym.N_ACTIONS} OBS_FLOATS={taverngym.OBS_FLOATS} OBS_CARDS={taverngym.OBS_CARDS}")

    t0 = time.perf_counter()
    game = taverngym.Game(seed=1)
    steps = 0
    while not game.is_over():
        for seat in range(game.num_players):
            if game.needs_action(seat):
                game.step(seat, game.bot_action(seat))
                steps += 1
    elapsed = time.perf_counter() - t0
    placements = list(game.placements())
    print(f"placements: {placements}")
    print(f"steps={steps} elapsed={elapsed:.2f}s")
    print("OK: bot game finished")


if __name__ == "__main__":
    main()
