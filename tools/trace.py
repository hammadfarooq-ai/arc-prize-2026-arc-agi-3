"""Trace the explorer's first N actions on one game.

usage: python tools/trace.py vc33 80
"""
import logging
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".pylibs"))
sys.path.insert(0, os.path.join(ROOT, "agent"))
logging.disable(logging.CRITICAL)
os.environ.setdefault("ONLY_RESET_LEVELS", "true")

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402
import importlib; Explorer = importlib.import_module(os.getenv("EXPL_MOD","explorer")).Explorer  # noqa: E402


def main():
    game, n = sys.argv[1], int(sys.argv[2])
    arc = Arcade(operation_mode=OperationMode.OFFLINE,
                 environments_dir=os.path.join(ROOT, "data", "environment_files"))
    gid = next(e.game_id for e in arc.get_environments() if e.game_id.startswith(game))
    env = arc.make(gid)
    ex = Explorer()
    r = env.reset()
    ex.observe(r.frame, r.state.name, r.levels_completed, r.available_actions)
    prev = np.array(r.frame[-1])
    for i in range(n):
        node = ex.nodes.get(ex.cur_key)
        aid, data = ex.next_action()
        r = env.step(GameAction.from_id(aid), data=data or {})
        f = np.array(r.frame[-1]) if len(r.frame) else prev
        d = f != prev
        ys, xs = np.nonzero(d)
        bb = (ys.min(), xs.min(), ys.max(), xs.max()) if len(ys) else None
        tier = int(node.tier[ex.prev_idx]) if (node is not None and ex.prev_idx is not None) else None
        print(f"{i:4d} a={aid} {data or ''} tier={tier} diff={int(d.sum())} bb={bb} "
              f"state={r.state.name} lv={r.levels_completed} nodes={len(ex.nodes)} vol={int(ex.vol.sum())} plan={len(ex.plan)}")
        ex.observe(r.frame if len(r.frame) else [prev], r.state.name, r.levels_completed, r.available_actions)
        prev = f


if __name__ == "__main__":
    main()
