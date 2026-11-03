"""Kaggle ARC-AGI-3 agent wrapper around the CPU graph explorer."""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Optional

import numpy as np
from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from .explorer import Explorer

logger = logging.getLogger(__name__)

# Whole-run wall clock budget (notebook limit is 9h; leave margin for setup
# and scorecard close).
_START_FILE = "/kaggle/working/start_time.txt"
try:
    with open(_START_FILE) as _fh:
        _T0 = float(_fh.read().strip())
except Exception:
    _T0 = time.time()
_BUDGET_S = float(os.getenv("AGENT_BUDGET_S", str(8 * 3600 + 5 * 60)))
_DEADLINE = _T0 + _BUDGET_S

_quiet_once = threading.Lock()
_quieted = [False]


def _quiet_logs():
    # The framework logs every action at INFO; with ~100 games that is
    # millions of lines. Keep warnings only.
    with _quiet_once:
        if not _quieted[0]:
            logging.getLogger().setLevel(logging.WARNING)
            _quieted[0] = True


class MyAgent(Agent):
    MAX_ACTIONS = 10 ** 9

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _quiet_logs()
        self.ex = Explorer(seed=abs(hash(self.game_id)) % (2 ** 31))
        self._pending: Optional[dict] = None
        self._fail = 0

    # no per-frame recordings / unbounded frame history (memory + disk)
    def start_recording(self) -> None:  # noqa: D401
        return None

    def append_frame(self, frame: FrameData) -> None:
        self.frames = [frame]
        if frame.guid:
            self.guid = frame.guid

    def is_done(self, frames, latest_frame) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        if time.time() > _DEADLINE:
            return True
        if self._fail > 50:
            return True
        return False

    def choose_action(self, frames, latest_frame) -> GameAction:
        try:
            fr = latest_frame.frame
            if fr is not None and len(fr) > 0:
                arr = np.asarray(fr[-1], dtype=np.int8)
            else:
                arr = self.ex.cur_frame if self.ex.cur_frame is not None else np.zeros((64, 64), np.int8)
            st = latest_frame.state.name if hasattr(latest_frame.state, "name") else str(latest_frame.state)
            self.ex.observe(arr, st, int(latest_frame.levels_completed or 0),
                            latest_frame.available_actions or [])
            aid, data = self.ex.next_action()
        except Exception as e:  # never crash the thread
            logger.warning(f"{self.game_id} explorer error: {e!r}")
            aid, data = 0, None
            if getattr(latest_frame, "state", None) not in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                aid = 1
        self._pending = data
        return GameAction.from_id(aid)

    def do_action_request(self, action: GameAction) -> FrameData:
        # Use our own per-agent payload instead of the (shared, mutable)
        # enum member's action_data: with ~100 threads that is a data race.
        data = dict(self._pending) if self._pending else {}
        raw = None
        for attempt in range(6):
            try:
                raw = self.arc_env.step(action, data=data, reasoning={})
            except Exception as e:  # noqa: BLE001
                logger.warning(f"{self.game_id} step error: {e!r}")
                raw = None
            if raw is not None:
                break
            time.sleep(min(2 ** attempt, 20))
        if raw is None:
            self._fail += 1
            raise ValueError("step failed")
        self._fail = 0
        return self._convert_raw_frame_data(raw)

    def _convert_raw_frame_data(self, raw) -> FrameData:
        if raw is None:
            return FrameData(levels_completed=0)
        return super()._convert_raw_frame_data(raw)

    def take_action(self, action: GameAction) -> Optional[FrameData]:
        try:
            return super().take_action(action)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"{self.game_id} take_action error: {e!r}")
            return None
