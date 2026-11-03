# ARC-AGI-3 plan, 2026-10-16 → 2026-11-02

## Realistic targets
- 1st place (Tufa 68.91, Chen 67.49) and the top 10 (~45+) are out of reach. Their methods are not public; the gap (≈ +40) is far beyond any harness tweak.
- Goal: a medal on the private board. Today's public lines: bronze ≈ 30.2, silver ≈ 31.6, gold ≈ 37.1.
- Public scores are each team's best draw. On the private board the dfranzen copies fall back to their mean (≈ 26). A real gain of +3 to +5 in expected score is what moves us up there.

## Constraints
- 1 submission per day: 17 left.
- 2 final selections.
- RTX Pro quota: 30 h/week. That is about 3 full evals (8.5 h each), shared with my other competitions.
- A difference between two evals has sd ≈ 3.2 public points. A single LB draw has sd ≈ 3.9.

## Levers, by expected value
1. **More useful tokens.** Giving every game 10% more tokens adds ≈ +4 public points.
   - a. Output cap and loop guard: 6% of generated tokens are in replies that hit the 12,288 cap. Cheap to build.
   - b. Speculative decoding: SPEC_STEPS=4 (already built), draft model swap (RadixArk NVFP4 MTP drafter, as sirikilohit used). Screen these in one throughput-benchmark kernel (several server configs in one ~2 h session) before spending 8.5 h evals on them.
   - c. NVFP4 target model (lordhansolo / RadixArk checkpoints, 1,135–1,159 peak tok/s vs our 946). Biggest upside but risky.
2. **Reasoning per token.** These are noise-level so far:
   - strategy audit (eval +1.2 ± 3.1, LB 28.90);
   - sirikilohit's pinned solved-level rules and winning actions;
   - temperature 0.6.
3. **Scheduling.** D' (LB 27.14, 29.25) is not evaluated offline yet.

## Schedule
- **Oct 16 (done):** built the loop guard (`--think-close`) and the throughput-benchmark kernel (`kaggle/bench-throughput`), both tested locally with mocks. The quota was still exhausted at 13:30 UTC.
- **Oct 17:** the user gave ARC first claim on the GPU quota this week.
  - Running in the background: push_queue for smoke + bench, plus `tools/chain_smoke_eval.sh`. The chain waits for the smoke run; if it passes (fired > 0, recovered ≥ 80%, failed 0), it queues `kaggle/eval-thinkclose` itself.
  - Logs: `runs/push_*.log` and `runs/chain_smoke_eval.log`. These stop if the Claude session closes, so check them at the start of each session.
- **Oct 18:**
  - eval-thinkclose finished at 47.56 (+2.8 ± 3.2 vs base) and was submitted as LB draw #1.
  - Running: bench v2 (bench v1 had landed on a T4) and eval-tc-audit, the combo.
  - Next eval: the bench winner if one is ≥ +10%; otherwise a repeat of think-close to firm up its evidence.
- **Oct 19:**
  - think-close LB #1 = 24.31 (noise-level low). Combo eval 46.40: the audit adds nothing (2×2 main effect 0.0), think-close +1.6 ± 2.3. Think-close is now positive in two evals (tc and tc+audit), which meets the "two evals both positive" rule, weakly.
  - Bench v2 measured only base (spec4 / nvdraft hit the startup deadline on slow relaunches). Bench v3 is built (35 min deadline, no base_repeat).
  - **Weekly GPU quota is used up until ~10-24 09:30 UTC.** No GPU work until then; only daily LB draws.
- **LB draws Oct 20–23 (no GPU):** 10-20 D' #3, 10-21 think-close #3, 10-22 combo #1, 10-23 D' #4 (or think-close #4 if D' has fallen back). D' has the best LB mean but no eval; think-close has the best eval but one low draw.
- **Oct 21:** think-close LB #2 = 27.44 (mean 25.88); D' #3 submitted. Colab G4 (same RTX PRO 6000) checked as extra GPU: if the hardware check passes, bench v3 runs on Colab before 10-24, and the Kaggle quota goes to evals only.
- **Oct 22–23:** D' #3 = 31.44 (mean 29.28 over 3 draws, the best LB evidence). Combo #1 = 0.00, a broken run (likely infrastructure); combo dropped. D' #4 submitted 10-23. From 10-24: alternate think-close and D' draws.
- **Oct 24:** think-close draw #3 submitted (04:05 UTC). Pushed in parallel at 04:05: bench v3 and Eval A = think-close + D' (`kaggle/eval-tc-dprime`).
- **Oct 25:** think-close #3 = 0.00, the second zero, and both zeros contain think-close: think-close is dropped from the LB. Eval A (tc+D') = 48.66, best eval. Bench v3 errored with an empty log. D' draw #5 submitted. **Remaining slots: D' every day** (best LB mean, never zero), unless a D'-only variant gets better evidence. **Final picks: D' + base** (or audit), both without think-close.
- **Oct 25 (later):** D' #5 also 0.00 → the zeros are Kaggle's slow model loading + scorecard expiry, not think-close. Built `--keepalive`. **From 10-26 submit only keepalive kernels:** tc+D'+keepalive (best eval 48.66), alternating with D'+keepalive. Final picks: tc+D'+KA and D'+KA.
- **Oct 27:** spec4 impossible, nvdraft model gone, NVFP4 skipped (user decision).
**Daily schedule (agreed 10-27; submit by hand at session start, background jobs die with the session):**
- Done: 10-28 tc+D'+KA (22.78) · 10-29 external (23.64) · 10-30 D'+KA #2 (ref 57806742). Next: 10-31 tc+D'+KA (`arc3-sub-tc-dprime-ka` v1) · 11-01 D'+KA (`arc3-sub-dprime-ka` v1) · 11-02 tc+D'+KA, then select finals tc+D'+KA and D'+KA.
- No NVFP4 work (user decision 10-27).
- Running: `arc3-eval-tc-dprime-ka` v1 (full eval, pushed 10-27 ~05:30 UTC, done ~14:00 UTC): a repeat of tc+D' (eval 48.66) plus keepalive, to firm up the final pick. Report: `python tools/eval_report.py malikhammadfarooq/arc3-eval-base malikhammadfarooq/arc3-eval-thinkclose malikhammadfarooq/arc3-eval-tc-dprime malikhammadfarooq/arc3-eval-tc-dprime-ka`.
- Final picks (Nov 1-2): tc+D'+KA and D'+KA. If the repeat eval puts tc+D' clearly below think-close alone, reconsider pick 1.
- **Week of Oct 24 (30 h, ≈ 3 evals):**
  1. Bench v3 first (~2.2 h): `bash tools/push_queue.sh kaggle/bench-throughput`.
  2. Eval A: think-close + bench winner (if ≥ +10%); else think-close + D' (D' never had an offline eval).
  3. Eval B: repeat of whichever of A / think-close looks best, to firm up the final pick.
  - Nothing that finishes after ~10-31 can get LB draws, so the last eval of the week is evidence for the final pick only.
- **Week of Oct 17 (done; ≈ 3 evals used):**
  1. As soon as the quota is back, run these two in parallel (2 GPU slots, if the other competitions leave them free):
     - `kaggle/smoke-thinkclose` (~25 min);
     - `kaggle/bench-throughput` (~2 h).
  2. Eval: base + `--think-close` + the best benchmark config (if any is ≥ +10%, via `--serve`).
     Build it with `make_franzen_fork.py ... --full-eval --think-close [--serve SPEC_STEPS=4]`. The nvdraft config needs launcher edits, not just `--serve`.
  3. Eval: combo = winner of (2) + audit.
- NVFP4 target model: try only if a later benchmark shows ≥ +15% tok/s (not in bench v3).
- **Every day:** submit the variant with the best evidence; alternate repeats so each candidate gets 3+ LB draws.
- **Nov 1–2:** final selection.
  - Pick 1: the variant with the best pooled evidence (evals weighted over LB draws).
  - Pick 2: the safest strong variant (base or audit), so one bad bet cannot sink us.

## Decision rule
Adopt a change only if it shows **one** of:
- tok/s up ≥ 10% with the eval score not lower; or
- eval up ≥ +4 (≈ 1.3 sd), or two evals both positive.

Otherwise keep the simpler version.

## Final picks (user approved 10-31)
- D' 31.44 (ref 57471001) + tc+D'+KA 29.35 (ref 57650306). Swap in a new sub only if it beats 31.44 publicly. 11-01/11-02: D'+KA (`arc3-sub-dprime-ka` v1).
