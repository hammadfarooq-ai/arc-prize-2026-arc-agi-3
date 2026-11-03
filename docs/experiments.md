# ARC-AGI-3 experiment tracker

## Leaderboard draws (one competition rerun each; noise sd ≈ 3.9)
| date (UTC) | kernel / version | variant | public LB |
|---|---|---|---|
| 10-10 | arc3-franzen-m2 v2 | base (dfranzen unchanged) | 28.15 |
| 10-11 | arc3-franzen-m2-dprime v1 | D' slot priority | 27.14 |
| 10-14 | arc3-franzen-m2-dprime v1 (ref 57168899) | D' repeat | 29.25 |
| 10-15 | arc3-eval-audit v1 (ref 57201606) | strategy audit | 28.90 |
| 10-16 | arc3-eval-base v1 (ref 57269931) | base draw #2 | 23.45 |
| 10-17 | arc3-eval-audit v1 (ref 57305277) | audit draw #2 | 25.18 |
| 10-18 | arc3-eval-thinkclose v1 (ref 57333264) | think-close draw #1 | 24.31 |
| 10-19 | arc3-eval-thinkclose v1 (ref 57386465) | think-close draw #2 | 27.44 |
| 10-20 | — | slot missed (no session) | |
| 10-21 | arc3-franzen-m2-dprime v1 (ref 57471001) | D' draw #3 | 31.44 |
| 10-22 | arc3-eval-tc-audit v1 (ref 57511259) | combo draw #1 | **0.00** (status COMPLETE) |
| 10-23 | arc3-franzen-m2-dprime v1 (ref 57559038) | D' draw #4 | 27.56 |
| 10-24 | arc3-eval-thinkclose v1 (ref 57581721) | think-close draw #3 | **0.00** (status COMPLETE) |
| 10-25 | arc3-franzen-m2-dprime v1 (ref 57618459) | D' draw #5 | **0.00** (COMPLETE in < 45 min) |
| 10-26 | arc3-sub-tc-dprime-ka v1 (ref 57650306) | tc + D' + keepalive draw #1 | 29.35 (keepalive works: first non-zero since 10-23) |
| 10-27 | arc3-sub-dprime-ka v1 (ref 57689418) | D' + keepalive draw #1 | 24.00 |
| 10-28 | arc3-sub-tc-dprime-ka v1 (ref 57726054) | tc + D' + keepalive draw #2 | 22.78 |
| 10-29 | (not by me) fork of yongjiehishi/arc3-duck-lab-t06 as-is, ref 57751794, 00:01 UTC | "D' stack, no keepalive" | 23.64 |
| 10-30 | arc3-sub-dprime-ka v1 (ref 57806742, 13:24 UTC) | D' + keepalive draw #2 | 23.73 |
| 10-31 | arc3-sub-tc-dprime-ka-ple v2 (ref 57824171, 02:35 UTC) | tc + D' + keepalive + parallel PLE copy (draw #3 of tc+D'+KA) | 22.29 |
| 11-01 | arc3-sub-dprime-ka v1 (ref 57859766, 04:15 UTC) | D' + keepalive draw #3 | 31.47 (new best) |
| 11-02 | arc3-sub-tc-dprime-ka-slm v1 (ref 57887250, 00:00 UTC) | FINAL: tc + D' + keepalive + solved-level memory (eval 51.67) | 25.40 |

LB means so far (identical code scores 28.15 and 23.45, so a single draw tells little):
- base 25.80 (2 draws);
- D' 28.85 (4 draws: 27.14, 29.25, 31.44, 27.56); +3.1 vs base, diff sd ≈ 3.4, so ~0.9 sd: the best LB evidence so far, still not significant;
- audit 27.04 (2 draws);
- think-close 25.88 (2 draws: 24.31, 27.44); same as base.
- combo 0.00 (1 draw). **Not noise: the run broke.** Kaggle hides rerun logs, so the cause is unknown.
  - The notebook is the think-close cell plus the audit cell, each identical to a kernel that scored normally on the LB (24.31, 27.44; 28.90, 25.18). The combo's offline eval ran cleanly (46.40, no errors). So a code bug specific to the combo is unlikely but cannot be ruled out.
  - The likelier cause is an infrastructure failure in the rerun (server start, gateway, machine). That would mean any submission can score 0: so far 1 of 10 complete reruns. This is one more reason to pick 2 different, proven variants at the end.
  - Decision: drop the combo (the audit adds 0.0 in the 2×2 anyway); do not spend another slot to retest it.

**Bench `arc3-bench-spec4` v1 (10-26, configs spec4 then base):**
- **spec4 cannot run on this model:** the server fails at startup with `NotImplementedError: Qwen QSA requires speculative_num_draft_tokens <= the QSA compress ratio (4) ... got 5`. SPEC_STEPS=3 (4 draft tokens) is the maximum. spec4 and nvdraft_spec4 are dead; a spec4 submission would have scored 0.00. Kernel dir deleted.
- base (relaunch): startup 1531 s (slow mounts), tok/s@9+ 671.5 (v2: 679.6), accept 2.66, 0 retracts.
- Next: `arc3-bench-nvdraft` v1 (nvdraft only, SPEC_STEPS=3), pushed 10-26 ~16:30 UTC; compare with 671.5-679.6.

**10-29:** keepalive draws so far 29.35, 24.00, 22.78 (mean 25.4), below the pre-slowdown normal draws (~28). Possibly noise (sd 3.9), possibly the slow load (20-60 min) eating the rerun's 9 h budget, which keepalive cannot avoid. Today's slot was used at 00:01 UTC by a submission I did not make (team member or user), so D'+KA #2 was refused (400).

**10-30:** external 10-29 run (duck-lab-t06 fork, no keepalive) = 23.64, also in the low-20s band. Since the slowdown (10-24) every non-zero draw is 22.8-29.4 (mean 24.8), vs ~28 before: the slow load likely costs ~3 points for everyone, not just us. D'+KA #2 submitted.

**11-01:** D'+KA #3 = 31.47 (new best public). Final picks updated per the agreed rule: D'+KA 31.47 (ref 57859766) + D' 31.44 (ref 57471001). Last slot (11-02) is a free ticket: only selected if it beats these publicly. Candidate: solved-level memory (lalalia9, verbatim; author reports +2.8 LB; Book of Ideas: only text change with independent positive reports). Pushed 11-01 13:37 UTC: `arc3-eval-tc-dprime-slm` v1 (full eval, compare to tc+D' 48.66/45.86) and `arc3-sub-tc-dprime-ka-slm` v1 smoke. **Eval `arc3-eval-tc-dprime-slm` v1 = 51.67 (sd 2.08), +6.91 ± 3.2 vs base, +4.4 vs tc+D' mean of 2 (47.26); 43 won (tc+D' 39/32). Best eval so far.** SLM smoke (11-01 14:08 UTC): clean — patch on, levels pinned (1/3/5 levels), 125 OK requests, no errors. Rule: submit tc+D'+KA+SLM only if the smoke is clean and the eval is not below ~46; else D'+KA v1.

**10-31 13:20 UTC, leaderboard:** team ARCane Intelligence rank 493 / 5006 on public (best sub 31.44, D' 10-21). Lines now: silver (top 10%, rank 500) 31.41, bronze (rank 1001) 29.73, gold (~rank 20) 39.8; top 75.56 (private method). Post-slowdown draws (KA era): 29.35, 24.00, 22.78, 23.73, 22.29 (+ external 23.64): mean 24.3, vs pre-slowdown D' mean 28.85. A new draw now has expected ~24 and is unlikely to beat the old ones.
**Final-pick logic:** Kaggle scores private on the selected subs (best of 2); old subs keep their private score, and pre-slowdown runs had ~40 more minutes of play. Recommended picks: D' 31.44 (ref 57471001, 10-21) + tc+D'+KA 29.35 (ref 57650306, 10-26). These are also what Kaggle auto-selects (top 2 public) if nothing is ticked.

**10-31 research (public notebooks + forum):**
- Public "Score" on notebooks is a lucky draw. The top one, "The AGI Boys" (35.35), is dfranzen M2 with *no* change to the submission path (only the demo commit settings differ). duck-lab-t06 (33.78) = base + D'; our fork of it scored 23.64. The Book of Ideas (leoprovorov, 145 experiments) has 5 unchanged-base submissions: 31.73, 29.01, 25.37, 23.85, 28.04 (mean 27.6, sd 3.1).
- Jeroen Cottaar (2nd, discussion 753843): since ~10-24 every submission loses ~40-50 min *before the notebook starts*, and that time counts against the 9 h. Not fixable from the notebook. Explains why all post-10-24 draws (ours 22.8-29.4, mean ~24.7 over 5) sit ~3-4 below the earlier ~28.
- Leszek Gora (discussion 754206): with --ple-offload-embedding the PLE table copy is single-threaded page faults; splitting rows across 16 threads took ready time 23.1 -> ~10 min. We use --ple-offload-embedding with BF16 PLE, so it applies. Our old smoke: load_weight 604 s, scheduler_e2e 780 s, ready ~14.7 min.
- Built patches/ple_fast_copy.py (exact-match rewrite of qwen4_exp.py in the venv; no-op if the code differs; tested on the real wheel source + numeric equality test) and `--ple-fast` in make_franzen_fork.py. Kernels: arc3-sub-tc-dprime-ka-ple v1, arc3-sub-dprime-ka-ple v1 (smokes pushed 10-31 01:29 UTC).
- PLE smokes: v1 of both kernels did NOT patch (lib64 -> lib symlink gave two glob hits; fixed with realpath). v2 patched OK, server healthy, games played. Load times: unpatched 604 / 1463 / 307 s, patched 497 / 1366 s. In every run load_weight ~= the precache thread's full read of the 185 GB mount (+~30 s), so on our setup the bottleneck is mount throughput (137-600 MB/s, varies per machine), not the PLE copy. Verdict: patch is harmless, no measurable gain; kept in the 10-31 submission since the code path is otherwise identical.
- Not adopted: solved-level memory (Book: "Improvable", inside the noise band), 14 streams (Book: closed, LB 21.49), deadline from /proc/uptime (gain small, and a large loss if the 40-min theory is wrong).

**Repeat eval `arc3-eval-tc-dprime-ka` v1 (10-27/28) = 45.86 (sd 2.39), +1.09 ± 3.4 vs base; 32 won.** think-close fired 111 / recovered 111 / failed 0; D' replaced 100; keepalive 1 RESET over 187 s (load was slow in the commit run too).
- tc+D' over two evals: 48.66, 45.86 → **mean 47.26** (sd of mean ≈ 1.7), vs base 44.77 (my run) / 46.49 (dfranzen's): about +1.6 to +2.5, under 1.5 sd.
- Every eval with think-close (47.56, 46.40, 48.66, 45.86; mean 47.12) beats my base run; D' on top of think-close adds ≈ +0.1 (47.26 vs 47.56/46.40). Nothing here is significant on its own; together the evidence leans to think-close being a small real gain and D' being neutral offline.
- LB with the normal (non-zero) draws: D' 27.14, 29.25, 31.44, 27.56, 24.00 (D'+KA) → mean 27.88; tc+D'+KA 29.35 (1 draw); base 25.80 (2).
- Final picks stay: tc+D'+KA (pick 1) and D'+KA (pick 2).

**Bench `arc3-bench-nvdraft` v1 (10-27): no data.** `keithtyser/qwen3-8-flash-next-nvfp4` (the RadixArk checkpoint) did not mount; the API now returns 403 for it (deleted or private). NVIDIA's official `nvidia/qwen3-8-flash-next-nvfp4` is public, but its MTP head differs (FP8 block-scaled routed experts, not BF16), so the nvdraft code does not apply as is. The keithtyser NVFP4 public notebooks depend on the same missing model.
- Speed levers now: spec4 impossible (QSA limit), nvdraft source gone, NVFP4 target would need a new launcher for a 132 GB checkpoint on slow mounts, with ~5 days left. Judged not worth the 0.00 risk.

**Public LB 10-26 15:30 UTC (4,855 teams):** we (ARCane Intelligence) are rank 385 with best draw 31.44. Cutoffs: bronze (rank 485) 30.98, silver (rank 242) 32.29, gold (rank 19) 38.22.

**10-25 (later): the zeros are a Kaggle infrastructure problem, not think-close.** D' v1, which scored 27-31 four times, also scored 0.00 (finished in < 45 min).
- Forum (topics 751598, 751631, 751756, 752253): since ~10-20 model mounts read slowly; loading takes 20-60+ min instead of ~5. Kaggle staff confirmed "dataset/model cache instability" and are looking at rerunning affected submissions.
- **Mechanism:** the scorecard expires after ~15 min with no game action. dfranzen's notebook releases the harness after 12 min and sends ONE warmup RESET; with a 40 min load nothing follows it, so the run ends with 0.00. Also ARC3_HTTP_RETRY_INITIAL_SECONDS=900 lets each game wait only 15 min for the server.
- Zero dates 10-22, 10-24, 10-25 are all in the slow period; 10-21 and 10-23 were fine. The think-close theory below is withdrawn: tc+D' (eval 48.66) is the best candidate again.
- **Fix `--keepalive` (patches/keepalive.py):** the first game to start RESETs every 240 s while /health is not ready (max 3 h), and takes over the harness warmup; the HTTP grace becomes 7200 s. Mock test passed (ready after 1000 s → 5 RESETs, then normal play; other games unaffected); installs on the real harness and composes with D'.
- **Smoke `arc3-sub-tc-dprime-ka` v1 PASSED (RTX PRO 6000, 30 min):** model precache 580 s; at the 720 s deadline the server was still loading → `keepalive installed; server ready now: False`, 1 keepalive RESET, waited 177 s, then normal play (lp85 7/8 levels, vc33 4/7; ft09/ls20 never got one of the 2 smoke slots, as designed). think-close fired 1 / recovered 1; D' replaced 4. Submittable.
- `arc3-sub-dprime-ka` v1 pushed 07:48 UTC (smoke).
- **Smoke `arc3-sub-dprime-ka` v1 PASSED:** precache 845 s (slower); server not ready at 722 s → 2 keepalive RESETs over 432 s, then normal play (ft09 won 6/6). D' replaced 4. Submittable.
- The scheduled 00:05 submit died with the session (empty log); submitted by hand at 02:18 UTC instead. - Scheduled: `runs/submit_1026.sh` submits arc3-sub-tc-dprime-ka v1 at 10-26 00:05 UTC (background; dies if the session closes, log `runs/submit_1026.log`).
- Built 10-25: `arc3-sub-tc-dprime-ka` (tc + D' + keepalive, smoke commit run, pushed) and `arc3-sub-dprime-ka` (D' + keepalive, queued for a GPU slot).

**10-25 findings (morning; the think-close part is withdrawn, see above):**
- **Second 0.00, and both zeros contain think-close.** think-close draws: 24.31, 27.44, 0.00; combo (tc+audit): 0.00. The 8 draws without think-close never scored 0. If the 2 zeros fell on random draws, the chance that both land on the 4 tc draws is C(4,2)/C(13,2) ≈ 8%. Not proof, but think-close is now **unsafe for the LB** until the cause is known.
  - Offline, think-close never failed (88, 92, 108 fires; all recovered; 0 failed). Exceptions in the continuation are caught. A guess: a hidden-game edge case that crashes or hangs the SGLang server (e.g. continue_final_message near the context limit), which would kill every game. Rerun logs are hidden, so this cannot be checked.
  - Decision: no more think-close LB draws; it is not a final pick unless the cause is found and fixed.
- **Eval A `arc3-eval-tc-dprime` v1 = 48.66 (sd 2.31), +3.90 ± 3.3 vs my base, +1.1 vs think-close.** Best eval so far. 39 won (base 29, tc 31). tok/s@9+ 696.2. think-close fired 108, recovered 108, failed 0; D' replaced 100 fresh-game priorities. So D' adds about +1.1 on top of think-close offline (noise-level, but the same sign as its LB +3.1).
- **Bench v3: ERROR, no data.** The downloaded log is empty (`[]`), so it failed before printing anything. No speed lever was measured; spec4 / nvdraft stay untested.

**GPU runs pushed 10-24 04:05 UTC (quota back):**
- `arc3-bench-throughput` v3 (35 min startup deadline, configs base / spec4 / nvdraft / nvdraft_spec4; ~2.2 h). Metadata has `competition_sources`.
- `arc3-eval-tc-dprime` v1 = Eval A: think-close + D', `--full-eval` (~8.5 h). Checked before push: every code cell parses; the diff vs eval-thinkclose is only the D' cell, and vs franzen-m2-dprime only the full-eval settings + the think-close cell. The two patches touch disjoint code (think-close: `_chat_completion`, `build_chat_payload`; D': `priority_value`, `ProgressPace`, `_PriorityGate.acquire`, `_HarnessGameSession.play`).

**Colab as extra GPU (10-21, not tried yet):** Colab's "G4" runtime is the same RTX PRO 6000 (96 GB), ~8.7 compute units/h. A run there needs ≈ 205–210 GB of downloads (target 192 GB: main shards 84 + PLE 102 + extra 5; drafter 4.1; wheels; nvdraft mtp shards), host RAM ≥ 170 GB, driver ≥ 580. `notebooks/colab_runtime_check.ipynb` checks this in ~1 min; the download + bench notebook is built only if it passes.

The weekly GPU quota was back on 10-17 09:30 UTC. Both GPU slots were taken by other kernels at that time, so push_queue is waiting for a free slot.

## Offline full evaluations (commit run: 4 passes × 25 public games, ~8h10m, RTX Pro 6000)
sd of a 4-pass mean ≈ 2.3 (on a ~46 scale); a difference of two runs has sd ≈ 3.2.
Report: `python tools/eval_report.py dfranzen/arc-agi-3-milestone-2-solution malikhammadfarooq/arc3-eval-base ...`

| kernel | variant | status | score | tok/s@9+ | notes |
|---|---|---|---|---|---|
| dfranzen/arc-agi-3-milestone-2-solution v3 | base (dfranzen's own run, 10-03) | done | 46.49 | 692.6 | reference |
| arc3-eval-base v1 | base, my fork | done 10-14 | 44.77 (sd 2.41) | 694.4 | reproduces dfranzen within noise (−1.7 ± 3.3); 29 won |
| arc3-eval-audit v1 | + strategy-audit prompt at 40k tokens on a level, every 30k | done 10-14 | 45.99 (sd 1.92) | 695.5 | +1.2 ± 3.1 vs my base: noise. Fired 272×, no errors. Per-game swings large both ways (cn04/sp80/tu93 up, tn36/su15/s5i5 down). Scheduled for LB submission 10-15 00:01 UTC |
| arc3-eval-dprime | + D' slot priority | not run: weekly GPU quota (30h) hit 10-14 16:18 | | | |
| arc3-eval-temp06 | temperature 0.7 → 0.6 | built, not queued | | | |
| arc3-eval-spec4 | SPEC_STEPS 3 → 4 | built, not queued | | | throughput probe (now covered by the benchmark kernel) |
| arc3-eval-thinkclose v1 | + loop guard `--think-close` (patches/think_close.py) | done 10-17 | **47.56** (sd 2.09) | 693.8 | **+2.79 ± 3.2 vs my base** (0.9 sd), +1.1 vs dfranzen's run. Best eval so far, but not yet significant. Fired 88×, recovered 88, failed 0. 31 won. Big per-game swings (vc33 +40, r11l +33, sp80 +27; cd82 −37, s5i5 −28). |
| arc3-eval-tc-audit v1 | think-close + strategy audit (combo) | done 10-18 | 46.40 (sd 2.26) | 691.9 | +1.63 ± 3.3 vs my base, −1.2 vs think-close alone: noise. Think-close fired 92, recovered 92, failed 0; audit fired as in eval-audit. 27 won. Swings again large (tn36 42→7, tr87 93→74, ar25 100→79; s5i5 5→42, m0r0 32→48). |

**2×2 reading of the four evals (base / audit / tc / tc+audit = 44.77 / 45.99 / 47.56 / 46.40):**
- main effect of think-close = (2.79 + 0.41) / 2 = **+1.6 ± 2.3**;
- main effect of the audit = (1.22 − 1.16) / 2 = **0.0 ± 2.3**;
- so the audit adds nothing measurable; think-close is the only candidate, and it is still under 1 sd.

**Smoke result (arc3-smoke-thinkclose v1, 10-17, PASSED):**
- With cap 2048: `think-close fired: 10 recovered: 10 failed: 0`.
- Every continuation came back `finish=tool_calls` with 1 tool call. The log has no Traceback or Error, and the game kept acting (266 actions).
- So the server side works on pennyroyal SGLang: continue_final_message, and the reasoning parser treating the prefix's `</think>` as the end of reasoning.
- The ls20 score of 8.61 is meaningless: the cap was set artificially low.

## GPU checks built 10-16 (not run: weekly quota)
| kernel | what | expected cost | pass criterion |
|---|---|---|---|
| arc3-smoke-thinkclose | 1 game, 12 min; `--think-close` with LOCAL_ANALYZER_MAX_OUTPUT=2048, so the close path fires often | ~25 min | log shows `think-close fired: N recovered: M` with M ≈ N, `failed: 0`, and the game keeps acting |
| arc3-bench-throughput | `tools/make_bench_kernel.py`; replays 10 recorded games (eval-base p0 request logs, from the first ≥40k-token prompt) against 5 server configs: base, spec4, nvdraft, nvdraft_spec4, base_repeat; 13 min each, first 3 min excluded | ~2 h | a config is worth an 8.5 h eval only if tok/s@9+ ≥ +10% over the mean of base and base_repeat, with no rise in retracts |

**Bench v1 (10-17) failed and gave no data.**
- Its metadata had no competition source, so Kaggle gave the session a **Tesla T4**, not the RTX Pro 6000. Every launch died on the 102 GB PLE host-table page-lock.
- **Lesson:** the RTX Pro 6000 is only granted to notebooks attached to the competition.
- **Fix in v2 (running since 10-18 01:40):** the competition source is attached, and the notebook asserts an RTX PRO 6000 at its start.
- Host RAM on the RTX Pro machine is 176.9 GB; ~106 GB is in use during a run.

**Bench v2 (10-18, RTX PRO 6000, 5716 s) — base measured, spec4 / nvdraft got no data:**

| config | ok | startup s | tok/s@9+ | accept len | client tok/s | reqs | retracts |
|---|---|---|---|---|---|---|---|
| base | yes | 552 | 679.64 | 2.65 | 607.3 | 192 | 0 |
| spec4 | no (deadline) | 1202 | | | | | |
| nvdraft | no (deadline) | 1201 | | | | | |
| nvdraft_spec4 | skipped | | | | | | |
| base_repeat | yes | 1161 | 679.83 | 2.65 | 639.1 | 199 | 0 |

- **The bench is very precise:** base vs base_repeat differ by 0.03% in tok/s@9+. A +10% (or even +3%) config would stand out clearly.
- Replay tok/s (680) is ~2% below the real eval runs (694), as expected from the recorded-reply prefixes.
- **Why spec4 and nvdraft failed:** not the configs. The first launch loaded the target weights in 298 s (the precache thread had warmed the page cache). Each relaunch took ~18 min, because `--weight-loader-drop-cache-after-load` empties the page cache. spec4 and nvdraft were at shard 35/38 when the 20 min startup deadline hit; base_repeat just made it (1147 s).
- **Fix for v3 (built 10-19, not pushed):** startup timeout 35 min, base_repeat dropped. Expected ~2.2 h. The push on 10-19 11:50 UTC was refused: **weekly GPU quota (30 h) reached**; it resets ~10-24 09:30 UTC.

Benchmark details:
- **Configs:** nvdraft = sirikilohit's draft, i.e. RadixArk checkpoint `mtp.*` tensors only, modelopt_fp4, draft MoE flashinfer_cutlass.
- **Local test:** the parser reproduces eval-base's 694.4 tok/s@9+ from its serve.log, and a mock-server replay ran without errors.

## Loop-guard findings (10-16, local analysis)
- **`reasoning_effort` is honoured.** The model's chat_template.jinja accepts xhigh (default), medium and low, and SGLang passes chat_template_kwargs through.
  - **Problem:** the effort instruction is the first line of the system prompt. So every switch of `ARC3_REASONING_EFFORT_LADDER` breaks that game's whole prefix cache: a re-prefill of ~90k tokens at ~11.7k tok/s, twice per truncation (step down, then reset).
  - **Verdict:** the ladder is not worth it.
- **The capped replies are not repetition loops.** I read s5i5 p0 of eval-base: 5 of its 52 replies were capped at 12,288 tokens.
  - All were reasoning-only (~38k characters), with zlib ratio 0.34 vs 0.39–0.48 for normal long traces.
  - They are long deliberation that circles hypotheses ("Hmm, ..."). A repetition detector would not fire.
  - The turn ends with no action, and with preserve_thinking the dead trace is replayed in later requests.
- **The output-length distribution has a spike at the cap.** In dfranzen's run, 245 replies were ≥ 8,192 tokens, 138 were ≥ 10,240, and 89 hit the cap.
  - Lowering the cap alone would also cut ~150 legitimate long replies. Not worth it on its own.
- **Loop guard chosen: think-close (budget forcing).**
  - **Mechanism:** on a reasoning-only `finish_reason=length` reply, re-send with the trace + a short closing sentence + `</think>` as the final assistant message, with `continue_final_message=true` and max_tokens 4096.
  - **Server support:** pennyroyal SGLang 0.5.19 supports this.
    - It renders the prompt with the generation prompt and appends the prefix tokens, so the prompt is identical and the radix cache covers it.
    - Its qwen3 reasoning parser sees `</think>` in the prefix and parses the continuation as content and tool calls.
  - **Tests:** the mocked unit tests pass.
  - **Still unverified:** the server-side behaviour. That needs the smoke run.
- **The "NVFP4 MTP drafter" is really BF16.** The RadixArk checkpoint's quantization ignore list contains `mtp.*`, and its 31 mtp tensors are in model-bf16-0001{0,1,2}.
  - So it is likely a more accurate but larger draft than albucino's INT4 g32. That could mean a higher accept length but less KV. Real runs already reach KV token usage 0.99, so watch for retracts.

Queue pusher: `bash tools/push_queue.sh kaggle/eval-x kaggle/eval-y` (waits for a free GPU slot).
A finished eval kernel can itself be submitted (its competition path equals the submission path).

## Findings from dfranzen's run (serve.log + benchmark.json)
- Decode throughput is flat at 8–10 running requests (~682–696 tok/s): the GPU is saturated, so more streams (e.g. 12) should not add tokens.
- Prefill is ~2% of request time (93.6% prefix-cache hit); decode dominates.
- 89 requests hit the 12,288-token output cap (6% of all generated tokens).
- Score vs tokens: scaling every game's tokens 0.8 → 0.9 adds ~4 public points, so throughput / token efficiency is the main lever.
- Median level completes in ~20k generated tokens, 80% within 44k; first slot chunk per game ≈ 60k tokens.
