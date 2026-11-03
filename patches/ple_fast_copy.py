"""Parallel PLE row copy (from Leszek Gora's forum post, 2026-10-30, discussion 754206).

With --ple-offload-embedding, SGLang copies the PLE tables (~100 GB bf16) from a lazy
safetensors mmap into pinned host memory with one thread. That copy is a stream of page
faults from the input mount, one at a time, and dominates load time (reported 23 -> 10 min
to ready). Here each shard's row range is split across threads so the faults overlap.

patch(site_packages) rewrites sglang/srt/models/qwen4_exp.py in the venv before the server
starts. If the expected code is not found it changes nothing and returns False.
"""
import glob
import os

OLD = """                emb.weight.data[local_start : local_start + n_rows].copy_(
                    loaded_weight[src_start : src_start + n_rows].to(
                        device=emb.weight.device, dtype=emb.weight.dtype
                    )
                )
"""

NEW = """                _dst = emb.weight.data[local_start : local_start + n_rows]
                _src = loaded_weight[src_start : src_start + n_rows]
                import os as _os
                _nt = int(_os.environ.get("PLE_COPY_THREADS", "16"))
                if (_nt > 1 and _dst.device.type == "cpu" and _src.device.type == "cpu"
                        and _src.dtype == _dst.dtype and n_rows >= 65536):
                    from concurrent.futures import ThreadPoolExecutor as _Pool
                    _step = (n_rows + _nt - 1) // _nt
                    _spans = [(i, min(i + _step, n_rows)) for i in range(0, n_rows, _step)]

                    def _cp(ab, _d=_dst, _s=_src):
                        _d[ab[0]:ab[1]].copy_(_s[ab[0]:ab[1]])

                    with _Pool(max_workers=len(_spans)) as _ex:
                        list(_ex.map(_cp, _spans))
                else:
                    _dst.copy_(_src.to(device=emb.weight.device, dtype=emb.weight.dtype))
"""


def patch(root):
    hits = glob.glob(os.path.join(root, "**", "sglang", "srt", "models", "qwen4_exp.py"), recursive=True)
    hits = sorted({os.path.realpath(h) for h in hits})  # lib64 is a symlink to lib
    if len(hits) != 1:
        print(f"PLE_FAST: expected one qwen4_exp.py under {root}, found {hits}; unchanged", flush=True)
        return False
    path = hits[0]
    src = open(path, encoding="utf-8").read()
    if NEW in src:
        print("PLE_FAST: already patched", path, flush=True)
        return True
    if src.count(OLD) != 1:
        print(f"PLE_FAST: copy block found {src.count(OLD)} times in {path}; unchanged", flush=True)
        return False
    open(path, "w", encoding="utf-8").write(src.replace(OLD, NEW))
    for pyc in glob.glob(os.path.join(os.path.dirname(path), "__pycache__", "qwen4_exp.*.pyc")):
        os.remove(pyc)
    compile(open(path, encoding="utf-8").read(), path, "exec")
    print("PLE_FAST: patched", path, flush=True)
    return True
