# ARC Prize 2026 · ARC-AGI-3: from a CPU explorer to a measured LLM agent

My work for the Kaggle competition [ARC Prize 2026 – ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) (team **ARCane Intelligence**). ARC-AGI-3 is a set of interactive, turn-based games with no instructions. An agent has to work out the rules by playing, then clear levels with as few actions as it can.

| | |
|---|---|
| **Best public LB score** | **31.47** (top ~10% of 5,000+ teams on the public board; silver line was 31.41) |
| **Best offline eval** | **51.67** on 4 passes × 25 public games, vs 44.77 for the unchanged baseline |
| **Daily submissions logged** | 28, each written up in [`docs/experiments.md`](docs/experiments.md) |
| **Hardware** | Kaggle RTX PRO 6000 (96 GB), 9 h per run, no internet |

---

## What's in this repo

This repo holds **only my own code and notes**. Third-party notebooks, the competition data, model outputs and other people's patches are left out on purpose. See [Credits](#credits) for where they come from.

```
.
├── agent/                 Phase 1: CPU-only graph-exploration agent (no LLM)
│   ├── explorer.py        state graph over masked frames, ranked candidates, BFS to frontier
│   └── my_agent.py        Kaggle Agent wrapper (thread-safe, retries, time budget)
├── patches/               Phase 2: patches I wrote for the LLM harness
│   ├── think_close.py     loop guard: close a reasoning trace cut at the token cap and act on it
│   ├── keepalive.py       stops 0.00 runs when Kaggle's model loading is slow
│   └── ple_fast_copy.py   multithreaded PLE embedding copy at model load (from a forum post)
├── tools/                 build, run, evaluate and benchmark tools
├── notebooks/             Colab runtime check (RTX PRO 6000, RAM, disk, driver)
└── docs/
    ├── experiments.md     every LB draw, offline eval, bench and finding, with dates
    └── plan.md            strategy, decision rules, final-pick logic
```

---

## Phase 1: a CPU agent without an LLM (Sept)

The first approach treated each game as graph search. It needs no GPU.

- **State graph.** Each distinct frame is a node. Step counters and energy bars are masked out first, so the same position always maps to the same node.
- **Candidate actions.** Each node keeps a ranked list of moves: simple keys, plus one click per connected colour component.
- **Exploration.** Try the best untested move. When a node runs out, walk by BFS (over known edges and RESET) to the frontier node with the cheapest untested move.
- **Tooling.** Offline scoring with the official formula ([`tools/run_local.py`](tools/run_local.py)), a local competition-mode gateway ([`tools/serve.py`](tools/serve.py)), an end-to-end test of the Kaggle rerun ([`tools/e2e.ps1`](tools/e2e.ps1)), and frame and trace viewers.

**Result:** about 0.2–0.3 on the LB. Blind search clears early levels in very few actions, but deep levels need the rules to be *understood*. That is what pushed the project to an LLM agent.

## Phase 2: an LLM agent, improved one measured change at a time (Oct)

The base was the public **dfranzen Milestone-2** solution: Tufa Labs' Duck harness, Qwen3.8-Flash-Next (INT4) with a speculative-decoding draft head, served by SGLang. Its code is not in this repo. I worked on top of it with a fork generator ([`tools/make_franzen_fork.py`](tools/make_franzen_fork.py)). It keeps the competition-rerun path byte-identical and adds each change as a separate, flag-controlled cell, so every variant can be built again and compared.

### My contributions

**1. Think-close loop guard** ([`patches/think_close.py`](patches/think_close.py))
- I read the request logs. 89 of 9,374 replies hit the 12,288-token output cap: 6% of all tokens, with no action at the end.
- The capped replies were not repetition loops (zlib ratio 0.34, vs 0.39–0.48 for normal long traces). They were long deliberation that kept circling the same ideas.
- The fix: close the trace with `</think>` and continue the same message (`continue_final_message`), so the turn still acts. The prompt stays identical, so the prefix cache still hits.
- Offline: **+2.8** over base. On the GPU it recovered 108 of 108 capped replies.

**2. Keepalive against 0.00 runs** ([`patches/keepalive.py`](patches/keepalive.py))
- From late October, model loading on Kaggle took 20–60+ min. With no game action for ~15 min, the scorecard expired and the run scored **0.00**. Three of my reruns in a row scored 0.00.
- The fix: the first game sends a RESET every 4 min until the model server's `/health` is ready, then plays normally.
- No 0.00 run after the fix.

**3. Parallel PLE copy at load** ([`patches/ple_fast_copy.py`](patches/ple_fast_copy.py))
- A forum post showed that the per-layer embedding copy runs as single-threaded page faults. I turned that idea into a safe, exact-match rewrite of the served model file.
- Tested on the real wheel source, plus a numeric equality test.
- On our setup, load time followed the input mount's read speed (307–1463 s across machines), so the fix gave no measurable gain. Written up as a negative result.

**4. Measurement tooling**
- Offline evaluation reports with per-run standard errors ([`tools/eval_report.py`](tools/eval_report.py)).
- A throughput benchmark that replays recorded requests against several server configs ([`tools/make_bench_kernel.py`](tools/make_bench_kernel.py)).
- A GPU queue that respects the 2-session limit ([`tools/push_queue.sh`](tools/push_queue.sh)), and an automatic smoke-then-eval chain ([`tools/chain_smoke_eval.sh`](tools/chain_smoke_eval.sh)).

**5. Integration and testing of public ideas**
- D' slot priority (shiiin9), strategy-audit prompt (lordhansolo), solved-level memory (lalalia9).
- Each one was ported, smoke-tested on the GPU and run through a full offline eval before it got a submission.

### Offline evaluation (4 passes × 25 public games)

| Variant | Score | vs base |
|---|---:|---:|
| Base (dfranzen M2, unchanged) | 44.77 | — |
| + strategy audit | 45.99 | +1.2 |
| + think-close | 47.56 | +2.8 |
| + think-close + D' | 48.66 / 45.86 | +3.9 / +1.1 |
| **+ think-close + D' + solved-level memory** | **51.67** | **+6.9** |

The difference between two eval runs has sd ≈ 3.2, so only the bottom row stands clearly above the noise.

---

## What I learned

1. **One leaderboard score is one random draw.** The same code scored 28.15 and 23.45, and a single draw has sd ≈ 3.9. The top public notebook (35.35) turned out to be the unchanged baseline on a lucky draw. Every decision here uses means over several runs and states its noise.
2. **Infrastructure can matter more than the model.** Late in the competition, every run lost ~40 min before the notebook even started (finding by Jeroen Cottaar, Tufa Labs), on top of slow model loads. Scores dropped ~3–4 points for everyone. The useful fixes were operational: keepalive, and judging runs by their load time.
3. **Read the logs before adding features.** The think-close guard came from reading what the model did when it ran out of tokens. It did not come from guessing.
4. **Write down negative results.** These changes looked good and did not pay off:
   - more parallel streams: +10% on the bench, but a worse leaderboard score
   - reasoning-effort switches: they break the prefix cache
   - the PLE fix on our setup
   - speculative decoding with 4 steps: the model refuses it
   All of them are in [`docs/experiments.md`](docs/experiments.md).
5. **Pick finals as a portfolio.** Private LB takes the best of two picks. I picked the two strongest proven runs over the newest variant, and kept a fresh draw only if it beat them on the public board.

---

## Reproducing

**CPU agent (Phase 1).** Download the competition data into `data/` and install its wheels into `.pylibs/`:

```bash
kaggle competitions download -c arc-prize-2026-arc-agi-3 -p data
```
```bash
pip install --target .pylibs data/arc_agi_3_wheels/*.whl
```
```bash
python tools/run_local.py --max-actions 2000 --jobs 4 --seed 1
```

**LLM agent (Phase 2).** Pull the public base notebook, then build a variant:

```bash
kaggle kernels pull dfranzen/arc-agi-3-milestone-2-solution -p ref/franzen -m
```
```bash
python tools/make_franzen_fork.py ref/franzen kaggle/my-variant "My Variant" --think-close --keepalive
```

Each GPU kernel's metadata needs `competition_sources: ["arc-prize-2026-arc-agi-3"]`. Without it, Kaggle assigns a T4 instead of the RTX PRO 6000. The flags `--dprime`, `--audit` and `--slm` need third-party patch files that are not included. The script says where to get them.

---

## Credits

Phase 2 is built on public work by others. Their code is **not** redistributed here:

- **Tufa Labs** (Jeroen Cottaar et al.): the Duck agent harness.
- **Daniel Franzen**: the Milestone-2 solution these experiments fork.
- **Intel**, **Albucino**, **John Pezzulli / Pennyroyal SGLang**: the quantized Qwen3.8-Flash-Next, its draft head, and the SGLang fork.
- **shiiin9** (D' priority), **lordhansolo** (strategy audit), **lalalia9 / rellik13 / sirikilohit** (solved-level memory), **Leszek Góra** (parallel PLE copy), **AlekseiProvorov** (the *Book of Ideas* experiment log): ideas used and credited in the code and notes.
- **ARC Prize Foundation** and **Kaggle**: the benchmark and the competition.
