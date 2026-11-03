"""Print a game's frame as hex characters, optionally after a list of actions.

usage: python tools/show.py vc33 "6:10,20 1 2"
"""
import logging
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".pylibs"))
logging.disable(logging.CRITICAL)

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402


def dump(f):
    print("    " + "".join(str(i % 10) for i in range(64)))
    for y, row in enumerate(f):
        print(f"{y:3d} " + "".join("0123456789abcdef"[v] for v in row))


def main():
    game = sys.argv[1]
    acts = sys.argv[2].split() if len(sys.argv) > 2 else []
    arc = Arcade(operation_mode=OperationMode.OFFLINE,
                 environments_dir=os.path.join(ROOT, "data", "environment_files"))
    gid = next(e.game_id for e in arc.get_environments() if e.game_id.startswith(game))
    env = arc.make(gid)
    r = env.reset()
    for a in acts:
        if a.startswith("6:"):
            x, y = map(int, a[2:].split(","))
            r = env.step(GameAction.ACTION6, data={"x": x, "y": y})
        else:
            r = env.step(GameAction.from_id(int(a)))
        print(f"after {a}: state={r.state.name} levels={r.levels_completed} nframes={len(r.frame)}")
    dump(np.array(r.frame[-1]))


if __name__ == "__main__":
    main()
