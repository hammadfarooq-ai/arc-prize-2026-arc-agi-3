import cProfile
import os
import pstats
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_local  # noqa: E402

game = sys.argv[1] if len(sys.argv) > 1 else "bp35-0a0ad940"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
cProfile.run(f"run_local.play('{game}', {n}, 600, 0)", "runs/prof.out")
p = pstats.Stats("runs/prof.out")
p.sort_stats("tottime").print_stats(18)
