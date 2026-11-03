"""Make a submit-ready fork of dfranzen's Milestone-2 notebook.

The competition rerun path is left byte-identical. Only the commit ("Save & Run")
branch is shortened: one public game, one pass, 12 minutes — enough to prove that the
patch applies, the server starts and a game is played, without burning ~8h of GPU quota.

usage: python tools/make_franzen_fork.py ref/franzen kaggle/franzen-m2 "ARC3 Franzen M2"
       (loop guard: --think-close; see patches/think_close.py)
       python tools/make_franzen_fork.py ref/franzen kaggle/eval-base "ARC3 Eval Base" --full-eval
       (variants: --env KEY=VAL, --serve KEY=VAL, repeatable)
       python tools/make_franzen_fork.py ref/franzen kaggle/franzen-m2-dprime "ARC3 Franzen M2 Dprime" --dprime

--dprime swaps the slot priority for the D' formula from shiiin9's public notebook
(patches/dprime_priority.py, embedded verbatim). It is active in the competition rerun.
"""
import glob
import json
import os
import re
import sys

# Third-party patches (not in this repo; get them from their public notebooks):
#   dprime_priority.py     - shiiin9/affectify-arc-31-54-in-a-single-sub
#   strategy_audit.py      - lordhansolo's Milestone-2 harness prompt
#   solved_level_memory.py - vickyfym/arc-agi-3-lalalia-final (lalalia9)
def _load_patch(name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patches", name)
    if not os.path.isfile(path):
        raise SystemExit(f"patches/{name} is third-party code and is not included in this repo; "
                         "see the comment above _load_patch for its public source.")
    return open(path, encoding="utf-8").read()


src_dir, out_dir, title = sys.argv[1], sys.argv[2], sys.argv[3]
DPRIME = "--dprime" in sys.argv[4:]
# --audit embeds patches/strategy_audit.py (lordhansolo's strategy-audit turn prompt,
# fired by tokens spent on the current level). Active in the competition rerun.
AUDIT = "--audit" in sys.argv[4:]
# --think-close embeds patches/think_close.py (loop guard: a reasoning-only reply cut at
# the output cap is closed with "</think>" and continued, so the turn still acts). Active in
# the competition rerun.
THINK_CLOSE = "--think-close" in sys.argv[4:]
# --full-eval keeps dfranzen's own commit run (4 passes x 25 public games, per-game budget
# scaled to the competition's): an ~8h offline A/B evaluation instead of a smoke test.
FULL_EVAL = "--full-eval" in sys.argv[4:]
# --env KEY=VAL overrides an harness env var after setup_env is applied (both run paths).
# --serve KEY=VAL replaces an entry of the sglang launcher's CFG dict, e.g. --serve MAXREQ=12.
ENV_OVERRIDES, SERVE_OVERRIDES = {}, {}
_args = sys.argv[4:]
for _flag, _target in (("--env", ENV_OVERRIDES), ("--serve", SERVE_OVERRIDES)):
    for _i, _a in enumerate(_args):
        if _a == _flag:
            _k, _v = _args[_i + 1].split("=", 1)
            _target[_k] = _v
user = "malikhammadfarooq"
# With D' the smoke test needs more games than slots, so the patched queue is exercised.
SMOKE_GAMES = ["ls20", "ft09", "vc33", "lp85"] if DPRIME else ["ls20"]
SMOKE_MINUTES, SMOKE_SLOTS = (18, 2) if DPRIME else (12, None)
GAMES = ("ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 "
         "sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30").split()

nb_path = glob.glob(os.path.join(src_dir, "*.ipynb"))[0]
nb = json.load(open(nb_path, encoding="utf-8"))
meta = json.load(open(os.path.join(src_dir, "kernel-metadata.json"), encoding="utf-8"))

excluded = [g for g in GAMES if g not in SMOKE_GAMES]
SMOKE_REPLACEMENTS = [
    ("demo_excluded_games = [] if TRUE_SUBMISSION else []\n",
     f"demo_excluded_games = [] if TRUE_SUBMISSION else {excluded!r}\n"),
    ("else:\n    if USE_PRIORITY_SCHEDULING:\n"
     "        bm.solver.concurrency = (25 - len(demo_excluded_games)) * bm.n_passes\n"
     "        bm.solver.max_runtime_s_per_game = 532*60 * bm.solver.concurrency // 110\n",
     "else:\n    bm.n_passes = 1  # fork: commit run is a short smoke test\n"
     "    if USE_PRIORITY_SCHEDULING:\n"
     "        bm.solver.concurrency = (25 - len(demo_excluded_games)) * bm.n_passes\n"
     f"        bm.solver.max_runtime_s_per_game = {SMOKE_MINUTES}*60\n"
     + (f"    os.environ['ARC3_MAX_ACTIVE_STREAMS'] = '{SMOKE_SLOTS}'\n" if SMOKE_SLOTS else "")),
]
REPLACEMENTS = ([] if FULL_EVAL else SMOKE_REPLACEMENTS) + [
    # Inputs may be mounted at different paths (API push vs. web editor): discover them.
    ("SERVED_MODEL_NAME = 'flashnext'\n", "SERVED_MODEL_NAME = 'flashnext'\n" + '''
def _find_input(default, name, marker=None):
    """Return `default` if mounted there, else the dir called `name` under /kaggle/input
    (descending to the shallowest subdir holding `marker`, for model version folders)."""
    if os.path.exists(default):
        return default
    hits = []
    for root, dirs, _ in os.walk('/kaggle/input'):
        depth = root.count(os.sep) - 2
        hits += [os.path.join(root, d) for d in dirs if d.lower() == name.lower()]
        if depth >= 4:
            dirs[:] = []
    for hit in sorted(hits, key=len):
        if marker is None:
            return hit
        for root, _, files in os.walk(hit):
            if marker in files:
                return root
    raise FileNotFoundError(f'input {name!r} not found under /kaggle/input (wanted {default})')

for _root, _dirs, _ in os.walk('/kaggle/input'):
    if _root.count(os.sep) <= 5:
        print('INPUT', _root, flush=True)
WHEELHOUSE_DIR  = _find_input(WHEELHOUSE_DIR, 'pennyroyal-v253')
MODEL_DIR       = _find_input(MODEL_DIR, 'intel-qwen3.8-flash-next-w4a16-autoround', 'config.json')
DRAFT_MODEL_DIR = _find_input(DRAFT_MODEL_DIR, 'albucino-qwen3-8-flash-next-drafter', 'config.json')
ORIG_BUNDLE_DIR = _find_input(ORIG_BUNDLE_DIR, 'taaf-kaggle-source-bundle-copy')
COMP_WHEELS     = _find_input('/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels', 'arc_agi_3_wheels')
print('RESOLVED', WHEELHOUSE_DIR, MODEL_DIR, DRAFT_MODEL_DIR, ORIG_BUNDLE_DIR, COMP_WHEELS, flush=True)
'''),
]
HARD_WHEELS = '"/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"'

done = [0] * len(REPLACEMENTS)
n_wheels = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    s = "".join(cell["source"])
    for i, (old, new) in enumerate(REPLACEMENTS):
        if old in s:
            assert s.count(old) == 1
            s = s.replace(old, new)
            done[i] += 1
    n_wheels += s.count(HARD_WHEELS)
    s = s.replace(HARD_WHEELS, "COMP_WHEELS")
    cell["source"] = s.splitlines(keepends=True)
    cell["outputs"] = []
    cell["execution_count"] = None
assert done == [1] * len(REPLACEMENTS), f"replacements applied: {done}"
assert n_wheels == 2, f"hardcoded wheel paths replaced: {n_wheels}"

ENV_ANCHOR = "os.environ.update({k: str(v) for k,v in setup_env.items()})"
if ENV_OVERRIDES:
    cells = [c for c in nb["cells"] if ENV_ANCHOR in "".join(c["source"])]
    assert len(cells) == 1, len(cells)
    s = "".join(cells[0]["source"])
    assert s.count(ENV_ANCHOR) == 1
    s = s.replace(
        ENV_ANCHOR, ENV_ANCHOR + f"\nos.environ.update({ENV_OVERRIDES!r})  # fork variant overrides\n"
        f"print('fork env overrides:', {ENV_OVERRIDES!r})")
    cells[0]["source"] = s.splitlines(keepends=True)
if SERVE_OVERRIDES:
    cells = [c for c in nb["cells"] if "CFG = dict(" in "".join(c["source"])]
    assert len(cells) == 1, len(cells)
    s = "".join(cells[0]["source"])
    for k, v in SERVE_OVERRIDES.items():
        s, n = re.subn(rf"^(    {re.escape(k)}=).*,$", rf"\g<1>{v},  # fork override", s, count=1,
                       flags=re.M)
        assert n == 1, f"serve key {k} not found in CFG"
    cells[0]["source"] = s.splitlines(keepends=True)

# --nvdraft swaps the albucino INT4 draft for the RadixArk checkpoint's mtp.* tensors (BF16,
# sirikilohit's draft), as in the "nvdraft" config of tools/make_bench_kernel.py. Active in the
# competition rerun.
NVDRAFT = "--nvdraft" in sys.argv[4:]
NV_MODEL = "keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"
NVDRAFT_PRELUDE = r'''
def _find_nvfp4_dir():
    for root, dirs, files in sorted(os.walk("/kaggle/input"), key=lambda t: len(t[0])):
        if ("nvfp4" in root.lower() and "hf_quant_config.json" in files
                and "model.safetensors.index.json" in files):
            return root
    raise FileNotFoundError("RadixArk nvfp4 checkpoint not found under /kaggle/input")


def prepare_mtp_only_view(nv_dir, work_root):
    """Draft folder with only the RadixArk checkpoint's mtp.* tensors: the draft loader keeps
    only names containing "mtp", so the other shards are never read."""
    import json as _json
    nv_dir, view = Path(nv_dir), Path(work_root) / "mtp-only-view"
    if view.exists():
        return view
    index = _json.loads((nv_dir / "model.safetensors.index.json").read_text())
    mtp = {k: f for k, f in index["weight_map"].items() if "mtp" in k}
    files = sorted(set(mtp.values()))
    assert len(mtp) == 31 and all(k.startswith("mtp.") for k in mtp) and len(files) <= 4, (len(mtp), files)
    view.mkdir(parents=True)
    for f in nv_dir.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            (view / f.name).symlink_to(f)
    for name in files:
        (view / name).symlink_to(nv_dir / name)
    (view / "model.safetensors.index.json").write_text(_json.dumps({**index, "weight_map": mtp}))
    print("MTP-ONLY DRAFT VIEW", view, len(mtp), "tensors in", files, flush=True)
    return view


NVFP4_DIR = _find_nvfp4_dir()
print("NVDRAFT from", NVFP4_DIR, flush=True)
'''
NVDRAFT_EDITS = [
    ("    DRAFT_VIEW = prepare_draft_view(MODEL_DIR, DRAFT_MODEL_DIR, PREFIX)",
     "    DRAFT_VIEW = prepare_mtp_only_view(NVFP4_DIR, PREFIX)  # fork: --nvdraft"),
    ('"--speculative-draft-model-quantization", "compressed-tensors", "--speculative-moe-runner-backend", "auto",',
     '"--speculative-draft-model-quantization", "modelopt_fp4", '
     '"--speculative-moe-runner-backend", "flashinfer_cutlass",'),
]
if NVDRAFT:
    cells = [c for c in nb["cells"] if "CFG = dict(" in "".join(c["source"])]
    assert len(cells) == 1, len(cells)
    s = "".join(cells[0]["source"])
    for old, new in NVDRAFT_EDITS:
        assert s.count(old) == 1, f"nvdraft anchor not found once: {old[:60]!r}"
        s = s.replace(old, new)
    anchor = "CFG = dict("
    s = s.replace(anchor, NVDRAFT_PRELUDE + "\n" + anchor, 1)
    cells[0]["source"] = s.splitlines(keepends=True)
    if NV_MODEL not in meta.setdefault("model_sources", []):
        meta["model_sources"] = meta["model_sources"] + [NV_MODEL]


def _code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


if DPRIME:
    module = _load_patch("dprime_priority.py")
    cell_src = (
        "# D' slot priority (priority = A*M*C + B*phi) from shiiin9's public notebook\n"
        "# https://www.kaggle.com/code/shiiin9/affectify-arc-31-54-in-a-single-sub, module embedded verbatim.\n"
        "# Replaces dfranzen's priority_value and prices unstarted games by the same formula.\n"
        "# Raises if the scheduler names it patches are missing.\n"
        "import sys as _fp_sys, types as _fp_types\n"
        f"_FP_SOURCE = {module!r}\n"
        "_fp_mod = _fp_types.ModuleType('ours_form_priority')\n"
        "_fp_sys.modules['ours_form_priority'] = _fp_mod\n"
        "exec(compile(_FP_SOURCE, 'ours_form_priority.py', 'exec'), _fp_mod.__dict__)\n"
        "import inference.agent.tool_agent as _fp_ta\n"
        "import inference.framework.solver as _fp_solver\n"
        "_fp_mod.install_d(_fp_ta, _fp_solver)\n"
    )
    run_idx = [i for i, c in enumerate(nb["cells"])
               if c["cell_type"] == "code" and "await bm.run(" in "".join(c["source"])]
    assert len(run_idx) == 1, run_idx
    nb["cells"].insert(run_idx[0], _code_cell(cell_src))
    # After the run, report how many unstarted games were queued by the formula.
    nb["cells"].insert(run_idx[0] + 2, _code_cell("print('D-prime fresh games priced by formula:', _fp_mod.FRESH)\n"))

if AUDIT:
    module = _load_patch("strategy_audit.py")
    cell_src = (
        "# Strategy-audit turn prompt (patches/strategy_audit.py, ported from lordhansolo's\n"
        "# Milestone-2 harness), triggered by tokens spent on the current level. Raises if the\n"
        "# ToolAgent names it relies on are missing.\n"
        "import sys as _sa_sys, types as _sa_types\n"
        f"_SA_SOURCE = {module!r}\n"
        "_sa_mod = _sa_types.ModuleType('strategy_audit')\n"
        "_sa_sys.modules['strategy_audit'] = _sa_mod\n"
        "exec(compile(_SA_SOURCE, 'strategy_audit.py', 'exec'), _sa_mod.__dict__)\n"
        "import inference.agent.tool_agent as _sa_ta\n"
        "_sa_mod.install(_sa_ta)\n"
    )
    run_idx = [i for i, c in enumerate(nb["cells"])
               if c["cell_type"] == "code" and "await bm.run(" in "".join(c["source"])]
    assert len(run_idx) == 1, run_idx
    nb["cells"].insert(run_idx[0], _code_cell(cell_src))
    nb["cells"].insert(run_idx[0] + 2, _code_cell("print('strategy audits fired:', _sa_mod.FIRED[0])\n"))

if THINK_CLOSE:
    module = _load_patch("think_close.py")
    cell_src = (
        "# Loop guard (patches/think_close.py): a reasoning-only reply cut at the output cap\n"
        "# is closed with </think> and continued (continue_final_message), so the turn acts.\n"
        "# Raises if the ToolAgent names it relies on are missing.\n"
        "import sys as _tc_sys, types as _tc_types\n"
        f"_TC_SOURCE = {module!r}\n"
        "_tc_mod = _tc_types.ModuleType('think_close')\n"
        "_tc_sys.modules['think_close'] = _tc_mod\n"
        "exec(compile(_TC_SOURCE, 'think_close.py', 'exec'), _tc_mod.__dict__)\n"
        "import inference.agent.tool_agent as _tc_ta\n"
        "_tc_mod.install(_tc_ta)\n"
    )
    run_idx = [i for i, c in enumerate(nb["cells"])
               if c["cell_type"] == "code" and "await bm.run(" in "".join(c["source"])]
    assert len(run_idx) == 1, run_idx
    nb["cells"].insert(run_idx[0], _code_cell(cell_src))
    nb["cells"].insert(run_idx[0] + 2, _code_cell(
        "print('think-close fired:', _tc_mod.FIRED[0], 'recovered:', _tc_mod.RECOVERED[0], "
        "'failed:', _tc_mod.FAILED[0])\n"))

# --slm embeds patches/solved_level_memory.py (lalalia9's solved-level memory, verbatim):
# a pinned "SOLVED LEVELS" block at the end of the system prompt, refreshed only on drains.
SLM = "--slm" in sys.argv[4:]
if SLM:
    module = _load_patch("solved_level_memory.py")
    run_idx = [i for i, c in enumerate(nb["cells"])
               if c["cell_type"] == "code" and "await bm.run(" in "".join(c["source"])]
    assert len(run_idx) == 1, run_idx
    nb["cells"].insert(run_idx[0], _code_cell(module))

# --keepalive embeds patches/keepalive.py: while the model server is still loading,
# the first game sends a RESET every 4 min so the scorecard does not expire, and every
# game's first request may wait up to 2 h for the server. Active in the competition rerun.
KEEPALIVE = "--keepalive" in sys.argv[4:]
if KEEPALIVE:
    module = _load_patch("keepalive.py")
    cell_src = (
        "# Keepalive (patches/keepalive.py): model loading on Kaggle got slow (20-60+ min);\n"
        "# with no game action for ~15 min the scorecard expires and the run scores 0.00.\n"
        "# The first game RESETs every 4 min until /health is ready. Raises if names are missing.\n"
        "import os as _ka_os, sys as _ka_sys, types as _ka_types\n"
        "_ka_os.environ['ARC3_HTTP_RETRY_INITIAL_SECONDS'] = '7200'\n"
        f"_KA_SOURCE = {module!r}\n"
        "_ka_mod = _ka_types.ModuleType('keepalive')\n"
        "_ka_sys.modules['keepalive'] = _ka_mod\n"
        "exec(compile(_KA_SOURCE, 'keepalive.py', 'exec'), _ka_mod.__dict__)\n"
        "import inference.framework.solver as _ka_solver\n"
        "_ka_mod.install(_ka_solver, f'http://127.0.0.1:{SERVED_MODEL_PORT}/health')\n"
        "print('keepalive installed; server ready now:', _ka_mod._ready(f'http://127.0.0.1:{SERVED_MODEL_PORT}/health'), flush=True)\n"
    )
    run_idx = [i for i, c in enumerate(nb["cells"])
               if c["cell_type"] == "code" and "await bm.run(" in "".join(c["source"])]
    assert len(run_idx) == 1, run_idx
    nb["cells"].insert(run_idx[0], _code_cell(cell_src))
    nb["cells"].insert(run_idx[0] + 2, _code_cell(
        "print('keepalive RESETs:', _ka_mod.RESETS[0], 'waited s:', round(_ka_mod.WAITED_S[0]))\n"))

# --ple-fast embeds patches/ple_fast_copy.py into the launcher: before the server starts,
# the venv's qwen4_exp.py gets a multi-threaded PLE row copy (load ~23 -> ~10 min reported).
# Changes nothing if the expected code is missing.
PLE_FAST = "--ple-fast" in sys.argv[4:]
if PLE_FAST:
    module = _load_patch("ple_fast_copy.py")
    LAUNCH_ANCHOR = "# ---- assemble Pennyroyal / AutoRound launch arguments ----"
    cells = [c for c in nb["cells"] if LAUNCH_ANCHOR in "".join(c["source"])]
    assert len(cells) == 1, len(cells)
    s = "".join(cells[0]["source"])
    assert s.count(LAUNCH_ANCHOR) == 1
    s = s.replace(LAUNCH_ANCHOR, (
        "# ---- parallel PLE row copy (patches/ple_fast_copy.py) ----\n"
        "import types as _pf_types\n"
        f"_PF_SOURCE = {module!r}\n"
        "_pf_mod = _pf_types.ModuleType('ple_fast_copy')\n"
        "exec(compile(_PF_SOURCE, 'ple_fast_copy.py', 'exec'), _pf_mod.__dict__)\n"
        "try:\n"
        "    _pf_mod.patch(VENV)\n"
        "except Exception as _pf_exc:\n"
        "    print('PLE_FAST: error, unchanged:', repr(_pf_exc), flush=True)\n\n"
    ) + LAUNCH_ANCHOR)
    cells[0]["source"] = s.splitlines(keepends=True)

slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
os.makedirs(out_dir, exist_ok=True)
json.dump(nb, open(os.path.join(out_dir, "notebook.ipynb"), "w", encoding="utf-8"), indent=1)
new_meta = {
    "id": f"{user}/{slug}",
    "title": title,
    "code_file": "notebook.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "dataset_sources": meta.get("dataset_sources", []),
    "kernel_sources": meta.get("kernel_sources", []),
    "competition_sources": meta.get("competition_sources", []),
    "model_sources": meta.get("model_sources", []),
    "docker_image": meta.get("docker_image"),
    "machine_shape": meta.get("machine_shape", "NvidiaRtxPro6000"),
}
json.dump(new_meta, open(os.path.join(out_dir, "kernel-metadata.json"), "w"), indent=1)
print("wrote", out_dir, "id", new_meta["id"])
