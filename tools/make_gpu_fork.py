"""Make a submit-ready fork of a public GPU notebook.

The commit ("Save & Run") skips the ~2h offline public-25 play and only writes the
placeholder submission; the competition rerun path is left untouched.

usage: python tools/make_gpu_fork.py ref/scott kaggle/duck-flashnext-fork "ARC3 Duck FlashNext Fork"
"""
import glob
import json
import os
import re
import sys

src_dir, out_dir, title = sys.argv[1], sys.argv[2], sys.argv[3]
user = "malikhammadfarooq"

nb_path = glob.glob(os.path.join(src_dir, "*.ipynb"))[0]
nb = json.load(open(nb_path, encoding="utf-8"))
meta = json.load(open(os.path.join(src_dir, "kernel-metadata.json"), encoding="utf-8"))

MARK = "# Play the benchmark; watchdog stop and teardown run even if it raises.\ntry:\n"
patched = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    s = "".join(cell["source"])
    i = s.find(MARK)
    if i < 0:
        continue
    j = s.find("\nfinally:\n", i)
    body = s[i + len(MARK):j + 1]
    body = "".join(("    " + ln if ln.strip() else ln) for ln in body.splitlines(keepends=True))
    guard = (
        "    if not TRUE_SUBMISSION and os.environ.get('FORK_SKIP_OFFLINE', '1') == '1':\n"
        "        # Fast commit: skip the offline public-25 play, write the placeholder only.\n"
        "        import pandas as pd\n"
        "        pd.DataFrame([['1_0', '1', True, 1]], columns=['row_id', 'game_id', 'end_of_game', 'score'])"
        ".to_parquet(WORKING_DIR / 'submission.parquet', index=False)\n"
        "        print('FORK: offline validation skipped; serving setup succeeded.', flush=True)\n"
        "    else:\n"
    )
    s = s[:i] + MARK + guard + body + s[j + 1:]
    cell["source"] = s.splitlines(keepends=True)
    cell["outputs"] = []
    patched += 1
assert patched == 1, f"expected to patch exactly one cell, patched {patched}"

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

# Competition data may be mounted at different paths: discover it instead of hardcoding.
HARD = '"/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"'
n_hard = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    s = "".join(cell["source"])
    if HARD in s:
        n_hard += s.count(HARD)
        s = s.replace(HARD, "COMP_WHEELS")
        cell["source"] = s.splitlines(keepends=True)
discover = (
    "import os, glob, subprocess\n"
    "print('INPUT:', os.listdir('/kaggle/input') if os.path.isdir('/kaggle/input') else None, flush=True)\n"
    "_cands = glob.glob('/kaggle/input/**/arc_agi_3_wheels', recursive=True)\n"
    "print('WHEEL DIRS:', _cands, flush=True)\n"
    "COMP_WHEELS = _cands[0] if _cands else '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels'\n"
    "try:\n"
    "    print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], capture_output=True, text=True).stdout, flush=True)\n"
    "except Exception as e:\n"
    "    print('nvidia-smi failed', e)\n"
)
nb["cells"].insert(0, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                       "source": discover.splitlines(keepends=True)})
print("replaced hardcoded wheel paths:", n_hard)

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
