"""Run the explorer on the 25 public games offline and compute the ARC-AGI-3 score.

usage: python tools/run_local.py [--max-actions N] [--seconds S] [--games ls20,ft09] [--jobs 4]
"""
import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".pylibs"))
sys.path.insert(0, os.path.join(ROOT, "agent"))

logging.disable(logging.CRITICAL)
os.environ.setdefault("ONLY_RESET_LEVELS", "true")


def level_score(baseline, actions):
    return min(1.15, (baseline / max(actions, 1)) ** 2)


def game_score(baselines, level_actions):
    num = den = 0.0
    for i, b in enumerate(baselines):
        w = i + 1
        den += w
        if i < len(level_actions):
            num += w * level_score(b, level_actions[i])
    return min(1.0, num / den) if den else 0.0


def play(game_id, max_actions, seconds, seed, module="explorer"):
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction
    import importlib
    Explorer = importlib.import_module(module).Explorer

    arc = Arcade(operation_mode=OperationMode.OFFLINE,
                 environments_dir=os.path.join(ROOT, "data", "environment_files"))
    info = next(e for e in arc.get_environments() if e.game_id == game_id)
    env = arc.make(game_id)
    ex = Explorer(seed=seed)
    fd = env.reset()
    ex.observe(fd.frame, fd.state.name, fd.levels_completed, fd.available_actions)
    t0 = time.time()
    n = 0
    level_marks = []
    last_level = fd.levels_completed
    last_mark = 0
    resets = 0
    while n < max_actions and time.time() - t0 < seconds:
        aid, data = ex.next_action()
        if aid == 0:
            resets += 1
        r = env.step(GameAction.from_id(aid), data=data or {})
        n += 1
        if r is None:
            break
        frame = r.frame if len(r.frame) else [ex.cur_frame]
        ex.observe(frame, r.state.name, r.levels_completed, r.available_actions)
        if r.levels_completed > last_level:
            level_marks.append(n - last_mark)
            last_mark = n
            last_level = r.levels_completed
        if r.state.name == "WIN":
            break
    sc = game_score(info.baseline_actions, level_marks)
    return {
        "game": game_id, "levels": len(level_marks), "of": len(info.baseline_actions),
        "score": round(sc * 100, 2), "actions": n, "resets": resets,
        "level_actions": level_marks, "baseline": info.baseline_actions[:len(level_marks) + 1],
        "secs": round(time.time() - t0, 1), "nodes": len(ex.nodes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-actions", type=int, default=3000)
    ap.add_argument("--seconds", type=float, default=240)
    ap.add_argument("--games", default="")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--module", default="explorer")
    args = ap.parse_args()

    games = sorted(d for d in os.listdir(os.path.join(ROOT, "data", "environment_files")))
    envdir = os.path.join(ROOT, "data", "environment_files")
    ids = []
    for g in games:
        for v in os.listdir(os.path.join(envdir, g)):
            ids.append(f"{g}-{v}")
    if args.games:
        pref = args.games.split(",")
        ids = [i for i in ids if any(i.startswith(p) for p in pref)]

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(play, g, args.max_actions, args.seconds, args.seed, args.module) for g in ids]
        for f in futs:
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                r = {"game": "?", "error": repr(e), "score": 0, "levels": 0}
            results.append(r)
            print(json.dumps(r), flush=True)
    tot = sum(r["score"] for r in results) / max(len(results), 1)
    lv = sum(r.get("levels", 0) for r in results)
    print(f"\nTOTAL score={tot:.3f}  levels={lv}  games={len(results)}  wall={time.time()-t0:.0f}s")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"total": tot, "results": results}, fh, indent=1)


if __name__ == "__main__":
    main()
