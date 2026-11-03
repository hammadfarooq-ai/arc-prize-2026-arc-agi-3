"""Build a throughput-benchmark kernel: one GPU session that serves several SGLang configs in
turn and measures decode tok/s and speculative accept length under the real workload.

The workload is replayed, not played.
- Each of 10 worker threads walks one public game's recorded request sequence, from the
  first request with a prompt of at least 40k tokens on. The sequences are the
  `*_p0_requests.jsonl` files in arc3-eval-base's output, attached as a kernel source.
- Every config gets the same prompts in the same order, so the configs differ only in the
  server.
- Generation is live (temperature 0.7, max_tokens 12288, the harness's sampling), so the
  output lengths and the draft acceptance are the model's own.
- The next prompt carries the recorded reply rather than the live one, so the prefix cache
  hits a little less often than in a real run. That is the same for every config.

Each config:
1. Start the server: dfranzen's launcher cell, run unchanged except for its CFG.
2. Replay for RUN_S seconds.
3. Stop the server.
4. From that config's own serve log, take the Decode batch lines after WARMUP_S. Report
   mean gen throughput (all lines, and lines with >= 9 running requests), mean accept
   length and token usage.

Results go to /kaggle/working/bench.json, and each serve log is kept.

usage: python tools/make_bench_kernel.py kaggle/bench-throughput "ARC3 Bench Throughput"
"""
import glob
import json
import os
import sys

out_dir, title = sys.argv[1], sys.argv[2]
# optional 3rd arg: comma list of configs to run, in that order (default: all)
ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
user = "malikhammadfarooq"
ref_nb = json.load(open(glob.glob(os.path.join(ROOT, "ref", "franzen", "*.ipynb"))[0], encoding="utf-8"))
ref_meta = json.load(open(os.path.join(ROOT, "ref", "franzen", "kernel-metadata.json"), encoding="utf-8"))
code = ["".join(c["source"]) for c in ref_nb["cells"] if c["cell_type"] == "code"]
precache_src = next(s for s in code if "precache_model_thread = threading.Thread(" in s)
serve_src = next(s for s in code if "CFG = dict(" in s)

# dfranzen's launcher, with its config, log path and draft choice opened up for the benchmark.
SERVE_EDITS = [
    ('LOG = str(Path(WORKING_DIR) / "serve.log")', "LOG = BENCH_LOG"),
    ("    SPEC_STEPS=3,\n)\n", "    SPEC_STEPS=3,\n)\nCFG.update(BENCH_CFG)\nprint('BENCH CFG', CFG, flush=True)\n"),
    ("precache_model_thread.start()", "_bench_start_precache()"),
    ("    DRAFT_VIEW = prepare_draft_view(MODEL_DIR, DRAFT_MODEL_DIR, PREFIX)",
     "    DRAFT_VIEW = (prepare_draft_view(MODEL_DIR, DRAFT_MODEL_DIR, PREFIX)\n"
     "                  if CFG.get('DRAFT_KIND', 'int4') == 'int4' else prepare_mtp_only_view(NVFP4_DIR, PREFIX))"),
    ('"--speculative-draft-model-quantization", "compressed-tensors", "--speculative-moe-runner-backend", "auto",',
     '"--speculative-draft-model-quantization", CFG.get("DRAFT_QUANT", "compressed-tensors"), '
     '"--speculative-moe-runner-backend", CFG.get("DRAFT_MOE", "auto"),'),
]
for old, new in SERVE_EDITS:
    assert serve_src.count(old) == 1, f"launcher anchor not found once: {old[:60]!r}"
    serve_src = serve_src.replace(old, new)

SETUP = r'''
import glob, json, os, re, signal, statistics, subprocess, threading, time
from pathlib import Path

WORKING_DIR = Path("/kaggle/working")
SERVED_MODEL_NAME = "flashnext"
SERVED_MODEL_PORT = 8001
# bench v2 (10-18): the first launch loaded its weights in 5 min (precached), every relaunch in
# ~18 min (the page cache is dropped after each load), so spec4 and nvdraft hit a 20 min deadline
SERVER_STARTUP_TIMEOUT = 50 * 60  # 10-26: model mounts read slowly; relaunches need more
NOTEBOOK_START_TIME = time.time()
BENCH_T0 = time.time()

# Stop at once on the wrong accelerator instead of burning quota on launches that cannot work.
_gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                      capture_output=True, text=True).stdout.strip()
print("GPU", _gpu, flush=True)
assert "RTX PRO 6000" in _gpu.upper() or "BLACKWELL" in _gpu.upper(), f"wrong GPU for this benchmark: {_gpu!r}"

def _find_dir(pred, what):
    for root, dirs, files in sorted(os.walk("/kaggle/input"), key=lambda t: len(t[0])):
        if pred(root, set(files)):
            return root
    raise FileNotFoundError(what)

for _root, _dirs, _ in os.walk("/kaggle/input"):
    if _root.count(os.sep) <= 5:
        print("INPUT", _root, flush=True)
WHEELHOUSE_DIR = _find_dir(lambda r, f: "requirements.lock" in f and os.path.isdir(os.path.join(r, "wheels")), "wheelhouse")
MODEL_DIR = _find_dir(lambda r, f: "config.json" in f and "autoround" in r.lower() and "drafter" not in r.lower(), "target")
DRAFT_MODEL_DIR = _find_dir(lambda r, f: "albucino" in r.lower() and "config.json" in f, "albucino draft")
try:
    NVFP4_DIR = _find_dir(lambda r, f: "nvfp4" in r.lower() and "hf_quant_config.json" in f and "model.safetensors.index.json" in f, "nvfp4")
except FileNotFoundError:
    NVFP4_DIR = None
REQUEST_FILES = sorted(glob.glob("/kaggle/input/**/*_p0_requests.jsonl", recursive=True))
print("RESOLVED", WHEELHOUSE_DIR, MODEL_DIR, DRAFT_MODEL_DIR, NVFP4_DIR, len(REQUEST_FILES), "request files", flush=True)
'''

HELPERS = r'''
def _bench_start_precache():
    if not getattr(_bench_start_precache, "done", False):
        _bench_start_precache.done = True
        precache_model_thread.start()


def prepare_mtp_only_view(nv_dir, work_root):
    """Draft folder with only the RadixArk checkpoint's mtp.* tensors (sirikilohit's M108):
    the draft loader keeps only names containing "mtp", so the other shards are never read."""
    nv_dir, view = Path(nv_dir), Path(work_root) / "mtp-only-view"
    if view.exists():
        return view
    index = json.loads((nv_dir / "model.safetensors.index.json").read_text())
    mtp = {k: f for k, f in index["weight_map"].items() if "mtp" in k}
    files = sorted(set(mtp.values()))
    assert len(mtp) == 31 and all(k.startswith("mtp.") for k in mtp) and len(files) <= 4, (len(mtp), files)
    view.mkdir(parents=True)
    for f in nv_dir.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            (view / f.name).symlink_to(f)
    for name in files:
        (view / name).symlink_to(nv_dir / name)
    (view / "model.safetensors.index.json").write_text(json.dumps({**index, "weight_map": mtp}))
    print("MTP-ONLY DRAFT VIEW", view, len(mtp), "tensors in", files, flush=True)
    return view


def gpu_used_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return int(out[0]) if out and out[0].strip().isdigit() else -1


def stop_server():
    p = globals().get("proc")
    if p is not None and p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(90)
        except Exception:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass
    subprocess.run("pkill -9 -f 'sglang serve' || true", shell=True)
    for _ in range(90):
        if 0 <= gpu_used_mb() < 2000:
            break
        time.sleep(2)
    print("server stopped; GPU used MB", gpu_used_mb(), flush=True)


def healthy():
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{SERVED_MODEL_PORT}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def launch(name, overrides):
    global BENCH_CFG, BENCH_LOG, NOTEBOOK_START_TIME
    BENCH_CFG, BENCH_LOG = dict(overrides), str(WORKING_DIR / f"serve-{name}.log")
    NOTEBOOK_START_TIME = time.time()
    exec(compile(SERVE_SRC, f"launcher-{name}", "exec"), globals())
    return healthy(), round(time.time() - NOTEBOOK_START_TIME)
'''

WORKLOAD = r'''
import requests

N_WORKERS, ROWS_PER_WORKER, MIN_PROMPT = 10, 60, 40000
GAMES = ["tu93", "s5i5", "ls20", "dc22", "ka59", "su15", "r11l", "re86", "sc25", "tn36",
         "tr87", "sb26", "m0r0", "vc33", "lf52"]


def load_sequence(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    seq, pending = [], None
    for r in rows:
        if r.get("event") == "request":
            pending = r
        elif r.get("event") == "response" and pending is not None:
            if (r.get("usage") or {}).get("prompt_tokens", 0) >= MIN_PROMPT or seq:
                msgs = [{k: v for k, v in m.items() if k != "_arc3_control"} for m in pending["messages"]]
                seq.append({"messages": msgs, "tools": pending["tools"],
                            "tool_choice": pending.get("tool_choice") or "auto"})
            pending = None
        if len(seq) >= ROWS_PER_WORKER:
            break
    return seq


by_game = {os.path.basename(p)[:4]: p for p in REQUEST_FILES}
SEQUENCES = []
for g in GAMES:
    if g in by_game and len(SEQUENCES) < N_WORKERS:
        s = load_sequence(by_game[g])
        if len(s) >= 20:
            SEQUENCES.append((g, s))
print("workload:", [(g, len(s)) for g, s in SEQUENCES], flush=True)
assert len(SEQUENCES) == N_WORKERS, "not enough recorded game sequences"


def replay(run_s):
    """N_WORKERS threads, each sending its game's requests one after another until run_s."""
    url = f"http://127.0.0.1:{SERVED_MODEL_PORT}/v1/chat/completions"
    stop_at = time.time() + run_s
    done = []
    lock = threading.Lock()

    def worker(game, seq):
        i = 0
        while time.time() < stop_at:
            row = seq[i % len(seq)]
            i += 1
            payload = {"model": SERVED_MODEL_NAME, "messages": row["messages"], "tools": row["tools"],
                       "tool_choice": row["tool_choice"], "max_tokens": 12288, "temperature": 0.7,
                       "top_p": 0.95, "top_k": 20, "stream": False,
                       "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}
            t = time.time()
            try:
                r = requests.post(url, json=payload, timeout=(10, max(5, stop_at - t + 5)))
                body = r.json()
                usage = body.get("usage") or {}
                fin = (body.get("choices") or [{}])[0].get("finish_reason")
                with lock:
                    done.append({"game": game, "t0": t, "t1": time.time(), "fin": fin,
                                 "prompt": usage.get("prompt_tokens", 0),
                                 "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                                 "out": usage.get("completion_tokens", 0)})
            except Exception as exc:
                if time.time() < stop_at:
                    with lock:
                        done.append({"game": game, "t0": t, "t1": time.time(), "fin": f"error: {exc}"[:200]})
                    time.sleep(2)

    threads = [threading.Thread(target=worker, args=gs, daemon=True) for gs in SEQUENCES]
    for th in threads:
        th.start()
    return threads, done


DEC = re.compile(r"Decode batch, #running-req: (\d+), #full token: (\d+), full token usage: ([\d.]+).*?"
                 r"accept len: ([\d.]+).*?gen throughput \(token/s\): ([\d.]+)")
PRE = re.compile(r"Prefill batch, #new-seq: \d+, #new-token: (\d+), #cached-token: (\d+)")


def parse_log(text):
    dec = [tuple(float(x) for x in m.groups()) for m in DEC.finditer(text)]
    pre = [tuple(int(x) for x in m.groups()) for m in PRE.finditer(text)]
    hi = [d for d in dec if d[0] >= 9]
    mean = lambda xs: round(statistics.mean(xs), 2) if xs else None
    return {"decode_lines": len(dec), "lines_9plus": len(hi),
            "tps_all": mean([d[4] for d in dec]), "tps_9plus": mean([d[4] for d in hi]),
            "accept_len": mean([d[3] for d in dec]), "accept_len_9plus": mean([d[3] for d in hi]),
            "running_mean": mean([d[0] for d in dec]), "token_usage_max": max([d[2] for d in dec], default=None),
            "prefill_new_tokens": sum(p[0] for p in pre), "prefill_cached_tokens": sum(p[1] for p in pre),
            "retracts": text.count("retract"), "kv_max_total": re.findall(r"max_total_num_tokens=(\d+)", text)[-1:]}
'''

RUN = r'''
RUN_S, WARMUP_S = int(os.environ.get("BENCH_RUN_S", 13 * 60)), int(os.environ.get("BENCH_WARMUP_S", 180))
NV = {"DRAFT_KIND": "nvfp4", "DRAFT_QUANT": "modelopt_fp4", "DRAFT_MOE": "flashinfer_cutlass"}
CONFIGS = [
    ("base", {}),
    ("spec4", {"SPEC_STEPS": 4}),
    ("nvdraft", NV),
    ("nvdraft_spec4", {**NV, "SPEC_STEPS": 4}),
    # base_repeat dropped in v3: in v2 it matched base to 0.03% (679.83 vs 679.64 tok/s@9+)
]
_ONLY = BENCH_ONLY_PLACEHOLDER
if _ONLY:
    CONFIGS = [next(c for c in CONFIGS if c[0] == n) for n in _ONLY]
print("CONFIGS", [c[0] for c in CONFIGS], flush=True)
RESULTS = []
for name, overrides in CONFIGS:
    if name.startswith("nvdraft") and NVFP4_DIR is None:
        RESULTS.append({"name": name, "error": "RadixArk checkpoint not attached"}); continue
    if name == "nvdraft_spec4" and not any(r["name"] == "nvdraft" and r.get("ok") for r in RESULTS):
        RESULTS.append({"name": name, "error": "skipped: nvdraft did not start"}); continue
    print(f"\n===== {name} {overrides} (elapsed {int(time.time() - BENCH_T0)}s) =====", flush=True)
    res, threads = {"name": name, "cfg": overrides}, []
    try:
        ok, startup_s = launch(name, overrides)
        res.update(ok=ok, startup_s=startup_s)
        if ok:
            threads, done = replay(RUN_S)
            time.sleep(WARMUP_S)
            off0 = Path(BENCH_LOG).stat().st_size
            time.sleep(max(0, RUN_S - WARMUP_S))
            off1 = Path(BENCH_LOG).stat().st_size
            with open(BENCH_LOG, "rb") as f:
                f.seek(off0)
                window = f.read(off1 - off0).decode("utf-8", "replace")
            res.update(parse_log(window))
            t_lo = time.time() - (RUN_S - WARMUP_S)
            fin = [d for d in done if d["t1"] >= t_lo and isinstance(d.get("out"), int)]
            res.update(requests_done=len(fin), client_out_tokens=sum(d["out"] for d in fin),
                       client_tps=round(sum(d["out"] for d in fin) / max(1, RUN_S - WARMUP_S), 1),
                       capped=sum(d["fin"] == "length" for d in fin),
                       cache_hit=round(sum(d["cached"] for d in fin) / max(1, sum(d["prompt"] for d in fin)), 3),
                       errors=sum(str(d["fin"]).startswith("error") for d in done))
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        stop_server()
        for th in threads:
            th.join(30)
    print("RESULT", json.dumps(res), flush=True)
    RESULTS.append(res)
    json.dump(RESULTS, open(WORKING_DIR / "bench.json", "w"), indent=1)

base = [r["tps_9plus"] for r in RESULTS if r["name"].startswith("base") and r.get("tps_9plus")]
ref = statistics.mean(base) if base else None
print(f"\n{'config':16s} {'ok':>3s} {'start_s':>7s} {'tps@9+':>7s} {'vs base':>7s} {'acc':>5s} {'client':>7s} {'reqs':>5s} {'kv':>8s}")
for r in RESULTS:
    tps = r.get("tps_9plus")
    rel = f"{(tps / ref - 1) * 100:+.1f}%" if tps and ref else ""
    print(f"{r['name']:16s} {str(r.get('ok', '-')):>3s} {str(r.get('startup_s', '-')):>7s} {str(tps):>7s} {rel:>7s} "
          f"{str(r.get('accept_len_9plus')):>5s} {str(r.get('client_tps')):>7s} {str(r.get('requests_done')):>5s} "
          f"{str(r.get('kv_max_total')):>8s} {r.get('error', '')}")
print("total elapsed", int(time.time() - BENCH_T0), "s")
'''


def cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [__doc__.split("\n\n")[0]]},
        cell(SETUP),
        cell(precache_src),
        cell(HELPERS + f"\nSERVE_SRC = {serve_src!r}\n"),
        cell(WORKLOAD),
        cell(RUN.replace("BENCH_ONLY_PLACEHOLDER", repr(ONLY))),
    ],
    "metadata": ref_nb.get("metadata", {}),
    "nbformat": 4, "nbformat_minor": 5,
}
slug = "".join(ch if ch.isalnum() else "-" for ch in title.lower()).strip("-")
while "--" in slug:
    slug = slug.replace("--", "-")
os.makedirs(out_dir, exist_ok=True)
json.dump(nb, open(os.path.join(out_dir, "notebook.ipynb"), "w", encoding="utf-8"), indent=1)
meta = {
    "id": f"{user}/{slug}", "title": title, "code_file": "notebook.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
    "enable_internet": False,
    "dataset_sources": ["dfranzen/pennyroyal-v253"],
    "kernel_sources": [f"{user}/arc3-eval-base"],
    # the RTX Pro 6000 is only granted to notebooks attached to the competition: without it
    # (bench v1, 10-17) the session ran on a small pre-sm80 GPU and every launch failed
    "competition_sources": ref_meta.get("competition_sources", ["arc-prize-2026-arc-agi-3"]),
    "model_sources": ref_meta["model_sources"] + ["keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"],
    "docker_image": ref_meta.get("docker_image"),
    "machine_shape": ref_meta.get("machine_shape", "NvidiaRtxPro6000"),
}
json.dump(meta, open(os.path.join(out_dir, "kernel-metadata.json"), "w"), indent=1)
print("wrote", out_dir, "id", meta["id"])
