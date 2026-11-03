"""Local competition-mode gateway serving the public games on :8001 (mimics Kaggle rerun)."""
import json
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, ".pylibs"))
logging.getLogger("werkzeug").setLevel(logging.ERROR)

from arc_agi import Arcade, OperationMode  # noqa: E402


def on_close(card):
    out = os.path.join(ROOT, "runs", "e2e_scorecard.json")
    with open(out, "w") as fh:
        fh.write(card.model_dump_json(indent=1))
    print(f"SCORECARD CLOSED score={card.score:.3f} -> {out}", flush=True)


if __name__ == "__main__":
    arc = Arcade(operation_mode=OperationMode.OFFLINE,
                 environments_dir=os.path.join(ROOT, "data", "environment_files"))
    arc.listen_and_serve(host="127.0.0.1", port=8001, competition_mode=True,
                         on_scorecard_close=on_close)
