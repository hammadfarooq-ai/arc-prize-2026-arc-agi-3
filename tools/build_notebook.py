"""Build the Kaggle submission notebook (CPU, no internet) from agent/*.py.

usage: python tools/build_notebook.py [--slug arc3-cpu-graph-explorer] [--title "..."]
Writes kaggle/<slug>/{notebook.ipynb, kernel-metadata.json}
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def code(src: str):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


def md(src: str):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="arc3-cpu-graph-explorer")
    ap.add_argument("--title", default="ARC3 CPU Graph Explorer")
    ap.add_argument("--user", default="malikhammadfarooq")
    args = ap.parse_args()

    explorer = open(os.path.join(ROOT, "agent", "explorer.py"), encoding="utf-8").read()
    my_agent = open(os.path.join(ROOT, "agent", "my_agent.py"), encoding="utf-8").read()

    cells = [
        md("# ARC-AGI-3 CPU graph-exploration agent\n"
           "No LLM, no GPU. Each distinct (status-bar-masked) frame is a node in a per-level state graph; "
           "the agent tries prioritised untested actions (keys + one click per colour component) and walks "
           "to the cheapest frontier state via BFS over known edges."),
        code("import time\nopen('/kaggle/working/start_time.txt', 'w').write(str(time.time()))"),
        code("!pip install --no-index --find-links /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels arc-agi python-dotenv"),
        code("%%writefile /kaggle/working/explorer.py\n" + explorer),
        code("%%writefile /kaggle/working/my_agent.py\n" + my_agent),
        code(r'''
import os, subprocess
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    !curl --fail --retry 999 --retry-all-errors --retry-delay 5 --retry-max-time 600 http://gateway:8001/api/games > /dev/null
    !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents /kaggle/working/ARC-AGI-3-Agents
    !cp /kaggle/working/my_agent.py /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py
    !cp /kaggle/working/explorer.py /kaggle/working/ARC-AGI-3-Agents/agents/templates/explorer.py
    with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
        f.write("""from typing import Type
from dotenv import load_dotenv
from .agent import Agent, Playback
from .swarm import Swarm
from .templates.random_agent import Random
from .templates.my_agent import MyAgent
load_dotenv()
AVAILABLE_AGENTS: dict[str, Type[Agent]] = {"random": Random, "myagent": MyAgent}
""")
    with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
        f.write("""SCHEME=http
HOST=gateway
PORT=8001
ARC_API_KEY=test-key-123
ARC_BASE_URL=http://gateway:8001/
OPERATION_MODE=online
RECORDINGS_DIR=/kaggle/working/server_recording
""")
    !cd /kaggle/working/ARC-AGI-3-Agents && MPLBACKEND=agg python main.py --agent myagent > /kaggle/working/run.log 2>&1; tail -n 50 /kaggle/working/run.log
'''),
        code(r'''
import os
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    import pandas as pd
    # Offline sanity check of the policy on the public games (short run)
    import sys, logging
    sys.path.insert(0, '/kaggle/working')
    logging.disable(logging.CRITICAL)
    try:
        import numpy as np
        from arc_agi import Arcade, OperationMode
        from arcengine import GameAction
        from explorer import Explorer
        arc = Arcade(operation_mode=OperationMode.OFFLINE,
                     environments_dir='/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files')
        for info in sorted(arc.get_environments(), key=lambda e: e.game_id)[:5]:
            env = arc.make(info.game_id); ex = Explorer(); fd = env.reset()
            ex.observe(fd.frame, fd.state.name, fd.levels_completed, fd.available_actions)
            for _ in range(300):
                a, d = ex.next_action()
                r = env.step(GameAction.from_id(a), data=d or {})
                ex.observe(r.frame if len(r.frame) else [ex.cur_frame], r.state.name, r.levels_completed, r.available_actions)
            print(info.game_id, 'levels', r.levels_completed, 'nodes', len(ex.nodes))
    except Exception as e:
        print('sanity check skipped:', repr(e))
    submission = pd.DataFrame(data=[['1_0', '1', True, 1]], columns=['row_id', 'game_id', 'end_of_game', 'score'])
    submission.to_parquet('/kaggle/working/submission.parquet', index=False)
'''),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
    out = os.path.join(ROOT, "kaggle", args.slug)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "notebook.ipynb"), "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1)
    meta = {
        "id": f"{args.user}/{args.slug}",
        "title": args.title,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "kernel_sources": [],
        "model_sources": [],
    }
    with open(os.path.join(out, "kernel-metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
