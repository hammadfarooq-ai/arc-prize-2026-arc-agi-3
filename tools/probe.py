"""Probe public games: available actions, what changes per action."""
import logging
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".pylibs"))

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402

logging.disable(logging.CRITICAL)


def last(fd):
    return np.array(fd.frame[-1])


def bbox(mask):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))


def main(filter_=None):
    arc = Arcade(operation_mode=OperationMode.OFFLINE,
                 environments_dir=os.path.join(ROOT, "data", "environment_files"))
    for info in sorted(arc.get_environments(), key=lambda e: e.game_id):
        if filter_ and not info.game_id.startswith(filter_):
            continue
        env = arc.make(info.game_id)
        fd = env.reset()
        f0 = last(fd)
        colors = np.bincount(f0.ravel(), minlength=16)
        print(f"== {info.game_id} avail={fd.available_actions} nframes={len(fd.frame)} "
              f"win_levels={fd.win_levels} colors={[(c, int(n)) for c, n in enumerate(colors) if n]}")
        prev = f0
        for a in fd.available_actions:
            if a == 6:
                continue
            r = env.step(GameAction.from_id(a))
            f = last(r)
            d = f != prev
            print(f"   A{a}: nframes={len(r.frame)} diff={int(d.sum())} bbox={bbox(d)} state={r.state.name}")
            prev = f


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
