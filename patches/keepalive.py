"""Keepalive for slow model loading in the competition rerun.

Since ~10-20 Kaggle's model mounts read slowly: loading takes 20-60+ min instead of ~5.
The scorecard expires if no game action arrives for ~15 min, and the run then scores
0.00 (our 10-22, 10-24, 10-25 reruns; many forum reports).

dfranzen's notebook releases the harness after 12 min and lets the first game send
one warmup RESET. With a long load nothing follows that RESET for the rest of the
load, so the scorecard still expires.

Here the first game to start keeps sending a RESET every INTERVAL_S while the model
server's /health endpoint is not ready, then plays normally. The extra RESETs come off
that one game's action count; that cost is small next to a 0.00 run.

install(solver_module, health_url) wraps _HarnessGameSession._play_inner. It raises
if the names it relies on are missing.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import urllib.request

log = logging.getLogger(__name__)

INTERVAL_S = float(os.environ.get("ARC3_KEEPALIVE_INTERVAL_S", "240"))
MAX_WAIT_S = float(os.environ.get("ARC3_KEEPALIVE_MAX_WAIT_S", str(3 * 3600)))
RESETS = [0]        # keepalive RESETs sent
WAITED_S = [0.0]    # seconds the keepalive game waited for the server
CLAIMED = [False]
_lock = threading.Lock()


def _ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def install(solver, health_url: str, *, ready=_ready, sleep=time.sleep,
            monotonic=time.monotonic) -> None:
    cls = getattr(solver, "_HarnessGameSession", None)
    if cls is None:
        raise RuntimeError("solver has no _HarnessGameSession")
    for name in ("_play_inner", "_execute_auto_reset", "should_stop"):
        if not callable(getattr(cls, name, None)):
            raise RuntimeError(f"_HarnessGameSession has no {name}")
    original = cls._play_inner
    if getattr(original, "_keepalive", False):
        return

    def _play_inner(self):
        if not ready(health_url):
            with _lock:
                mine = not CLAIMED[0]
                CLAIMED[0] = True
            if mine:
                # this game takes over the warmup role, so the harness's own single
                # warmup RESET is not added on top
                wlock = getattr(self.solver, "_warmup_lock", None)
                if wlock is not None:
                    with wlock:
                        self.solver._warmup_remaining = 0
                t0 = monotonic()
                run = getattr(self.game, "game_run", None)
                log.warning("%s: keepalive - server not ready, RESET every %.0fs until it is",
                            getattr(run, "game_id", "?"), INTERVAL_S)
                while not self.should_stop() and monotonic() - t0 < MAX_WAIT_S:
                    try:
                        self._execute_auto_reset()
                        RESETS[0] += 1
                    except Exception as exc:  # keep waiting; the next RESET may work
                        log.warning("keepalive RESET failed: %s", exc)
                    next_at = monotonic() + INTERVAL_S
                    while monotonic() < next_at and not ready(health_url):
                        sleep(5)
                    if ready(health_url):
                        break
                WAITED_S[0] = monotonic() - t0
                log.warning("keepalive done: %d RESETs over %.0fs, server ready=%s",
                            RESETS[0], WAITED_S[0], ready(health_url))
        return original(self)

    _play_inner._keepalive = True
    cls._play_inner = _play_inner
