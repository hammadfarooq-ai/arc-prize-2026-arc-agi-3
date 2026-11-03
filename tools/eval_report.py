"""Summarise full offline evaluations (4 passes x 25 public games) of dfranzen forks.

usage: python tools/eval_report.py dfranzen/arc-agi-3-milestone-2-solution malikhammadfarooq/arc3-eval-base ...

Downloads benchmark.json, summary.txt and serve.log of each kernel's latest run into
runs/eval/<slug>/ (skipped if already there), then prints per-run mean score, the
sampling sd of that mean (from the pass-to-pass spread within each game), tokens and
decode throughput, plus each run's difference to the first one with its sd.
"""
import json
import os
import re
import statistics as st
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "eval")
PATTERN = r"^(benchmark\.json|summary\.txt|serve\.log)$"


def fetch(ref):
    out = os.path.join(ROOT, ref.split("/")[1])
    if not os.path.exists(os.path.join(out, "benchmark.json")):
        os.makedirs(out, exist_ok=True)
        subprocess.run(["kaggle", "kernels", "output", ref, "-p", out, "--file-pattern", PATTERN],
                       check=False, capture_output=True)
    return out


def stats(d):
    b = json.load(open(os.path.join(d, "benchmark.json")))
    runs = b["game_runs"]
    by_game = defaultdict(list)
    for r in runs:
        by_game[r["game_id"][:4]].append(r["final_score"])
    games = sorted(by_game)
    mean = st.mean(st.mean(v) for v in by_game.values())
    var = sum(st.variance(v) / len(v) for v in by_game.values() if len(v) > 1) / len(games) ** 2
    tokens = sum(sum(h["generated_tokens"] or 0 for h in r["history"]) for r in runs)
    tps, acc = [], []
    log = os.path.join(d, "serve.log")
    if os.path.exists(log):
        for line in open(log, encoding="utf-8", errors="ignore"):
            m = re.search(r"#running-req: (\d+),.*accept len: ([\d.]+).*gen throughput \(token/s\): ([\d.]+)", line)
            if m and int(m.group(1)) >= 9:
                acc.append(float(m.group(2)))
                tps.append(float(m.group(3)))
    return dict(mean=mean, sd=var ** 0.5, n=len(runs), won=sum(r["state"] == "won" for r in runs),
                tokens=tokens / len(runs), tps=st.mean(tps) if tps else float("nan"),
                acc=st.mean(acc) if acc else float("nan"),
                games={g: st.mean(v) for g, v in by_game.items()})


def main(refs):
    res = [(ref, stats(fetch(ref))) for ref in refs]
    base = res[0][1]
    print(f"{'kernel':45s} {'score':>6s} {'sd':>5s} {'diff':>6s} {'runs':>4s} {'won':>3s} "
          f"{'tok/run':>8s} {'tok/s@9+':>8s} {'acc':>5s}")
    for ref, s in res:
        diff = s["mean"] - base["mean"]
        dsd = (s["sd"] ** 2 + base["sd"] ** 2) ** 0.5
        print(f"{ref:45s} {s['mean']:6.2f} {s['sd']:5.2f} {diff:+6.2f}+/-{dsd:.1f} {s['n']:4d} {s['won']:3d} "
              f"{s['tokens']:8.0f} {s['tps']:8.1f} {s['acc']:5.2f}")
    print("\nper game (mean over passes):")
    print("game " + " ".join(f"{ref.split('/')[1][-12:]:>12s}" for ref, _ in res))
    for g in sorted(base["games"]):
        print(f"{g:4s} " + " ".join(f"{s['games'].get(g, float('nan')):12.1f}" for _, s in res))


if __name__ == "__main__":
    main(sys.argv[1:])
