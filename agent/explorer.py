"""CPU graph-exploration policy for ARC-AGI-3.

The policy is framework independent: feed every observation to
`observe(frame, state, levels_completed, available_actions)` and ask for the
next move with `next_action()`, which returns (action_id, data_or_None).

Every distinct frame (with step counters / energy bars masked out) is a node
of a per-level state graph. A node owns a prioritised candidate list (simple
keys plus one click per connected colour component). We try the best untested
candidate of the current node; when it is exhausted (or only low-priority
candidates remain) we walk, by BFS over known edges and RESET, to the frontier
node whose best untested candidate is cheapest.

Memory matters (~100 games run concurrently), so per-node data lives in small
numpy arrays and frames are stored zlib-compressed.
"""
from __future__ import annotations

import random
import zlib
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import ndimage as _ndi
except Exception:  # pragma: no cover
    _ndi = None

RESET = 0
SIMPLE_IDS = (1, 2, 3, 4, 5, 7)
BORDER = 2           # pixels from the edge treated as status-bar band
N_TIERS = 5
TIER_COST = float(__import__("os").getenv("TIER_COST", "6.0"))      # one priority tier is worth this many walking steps
MAX_NODES = 60000    # per level safety cap
HEUR_W = float(__import__("os").getenv("HEUR_W", "1.0"))

UNTESTED, NOOP, MOVED, DEAD = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def _label(mask: np.ndarray):
    if _ndi is not None:
        return _ndi.label(mask)
    lab = np.zeros(mask.shape, np.int32)
    n = 0
    h, w = mask.shape
    for sy in range(h):
        for sx in range(w):
            if mask[sy, sx] and lab[sy, sx] == 0:
                n += 1
                stack = [(sy, sx)]
                lab[sy, sx] = n
                while stack:
                    y, x = stack.pop()
                    for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and lab[ny, nx] == 0:
                            lab[ny, nx] = n
                            stack.append((ny, nx))
    return lab, n


def components(frame: np.ndarray, ignore: np.ndarray):
    """List of (color, size, (y, x) representative, (y0, x0, y1, x1)) for each
    4-connected single-colour component, skipping ignored pixels. The
    representative is the member pixel closest to the component centroid."""
    out = []
    for c in np.unique(frame):
        m = (frame == c) & ~ignore
        if not m.any():
            continue
        lab, n = _label(m)
        if n == 0:
            continue
        ys, xs = np.nonzero(m)
        ls = lab[ys, xs]
        sizes = np.bincount(ls, minlength=n + 1)
        cy = np.bincount(ls, ys, minlength=n + 1) / np.maximum(sizes, 1)
        cx = np.bincount(ls, xs, minlength=n + 1) / np.maximum(sizes, 1)
        d = (ys - cy[ls]) ** 2 + (xs - cx[ls]) ** 2
        order = np.lexsort((d, ls))
        ls_o = ls[order]
        first = order[np.r_[True, ls_o[1:] != ls_o[:-1]]]
        y0 = np.full(n + 1, 99)
        x0 = np.full(n + 1, 99)
        y1 = np.full(n + 1, -1)
        x1 = np.full(n + 1, -1)
        np.minimum.at(y0, ls, ys)
        np.minimum.at(x0, ls, xs)
        np.maximum.at(y1, ls, ys)
        np.maximum.at(x1, ls, xs)
        for k in first:
            i = ls[k]
            out.append((int(c), int(sizes[i]), (int(ys[k]), int(xs[k])),
                        (int(y0[i]), int(x0[i]), int(y1[i]), int(x1[i]))))
    return out


# ---------------------------------------------------------------------------
# Graph structures
# ---------------------------------------------------------------------------

class Node:
    __slots__ = ("key", "zf", "acts", "tier", "sig", "status", "edges",
                 "dead", "ptr", "visits", "depth", "apos")

    def __init__(self, key: int, frame: np.ndarray, depth: int):
        self.key = key
        self.apos = None                         # avatar (y, x) if known
        self.zf = zlib.compress(frame.astype(np.int8).tobytes(), 1)
        self.acts = np.zeros((0, 3), np.int16)   # (action_id, x, y), priority order
        self.tier = np.zeros(0, np.int8)
        self.sig = np.zeros(0, np.int32)         # click signature id, -1 for keys
        self.status = np.zeros(0, np.int8)
        self.edges: Dict[int, int] = {}          # cand idx -> destination key
        self.dead: Dict[int, int] = {}           # cand idx -> times it ended the game
        self.ptr = 0
        self.visits = 0
        self.depth = depth

    @property
    def frame(self) -> np.ndarray:
        return np.frombuffer(zlib.decompress(self.zf), np.int8).reshape(64, 64)

    def best_untested(self) -> Optional[int]:
        st = self.status
        p = self.ptr
        n = len(st)
        while p < n and st[p] != UNTESTED:
            p += 1
        self.ptr = p
        return p if p < n else None

    def act(self, i: int) -> Tuple[int, int, int]:
        a = self.acts[i]
        return (int(a[0]), int(a[1]), int(a[2]))

    def index_of(self, action: Tuple[int, int, int]) -> Optional[int]:
        m = np.nonzero((self.acts[:, 0] == action[0]) & (self.acts[:, 1] == action[1])
                       & (self.acts[:, 2] == action[2]))[0]
        return int(m[0]) if len(m) else None


_YY, _XX = np.mgrid[0:64, 0:64]


def _color_stats(f: np.ndarray, vol: np.ndarray):
    m = ~vol
    v = f[m].astype(np.int64)
    cnt = np.bincount(v, minlength=16).astype(np.float64)
    sy = np.bincount(v, _YY[m], minlength=16)
    sx = np.bincount(v, _XX[m], minlength=16)
    return cnt, sy, sx


class AvatarModel:
    """Learns which colour is the player-controlled object and how each key
    displaces it, from per-colour centroid shifts between frames."""

    def __init__(self):
        self.rec: Dict[int, Dict[int, List[Tuple[float, float]]]] = {}
        self.color: Optional[int] = None

    def update(self, a: np.ndarray, b: np.ndarray, aid: int, vol: np.ndarray):
        ca, ya, xa = _color_stats(a, vol)
        cb, yb, xb = _color_stats(b, vol)
        moved = []
        for c in range(16):
            n0, n1 = ca[c], cb[c]
            if n0 < 1 or n1 < 1 or n0 > 1500:
                continue
            if abs(n0 - n1) > max(2.0, 0.15 * n0):
                continue
            dy = yb[c] / n1 - ya[c] / n0
            dx = xb[c] / n1 - xa[c] / n0
            if dy * dy + dx * dx >= 0.64:
                moved.append((c, dy, dx, n0))
        if not moved or len(moved) > 3:
            return
        for c, dy, dx, _ in moved:
            self.rec.setdefault(c, {}).setdefault(aid, []).append((dy, dx))
        best, best_n = None, 0
        for c, d in self.rec.items():
            n = sum(len(v) for v in d.values())
            if n > best_n:
                best, best_n = c, n
        if best_n >= 3:
            self.color = best

    def disp(self, aid: int) -> Optional[Tuple[float, float]]:
        if self.color is None:
            return None
        v = self.rec.get(self.color, {}).get(aid)
        if not v:
            return None
        arr = np.array(v[-8:])
        return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))

    def pos(self, f: np.ndarray, vol: np.ndarray):
        if self.color is None:
            return None
        m = (f == self.color) & ~vol
        n = int(m.sum())
        if n == 0 or n > 1500:
            return None
        return float(_YY[m].mean()), float(_XX[m].mean()), n


class Explorer:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        # cross-level statistics
        self.sig_ids: Dict[tuple, int] = {}
        self.sig_stats: List[List[int]] = []          # id -> [tries, effective]
        self.simple_stats: Dict[int, List[int]] = {}
        self.win_sigs: set = set()
        self.avatar = AvatarModel()
        self.visited_targets: set = set()
        self.vol =np.zeros((64, 64), bool)           # masked (counter) pixels
        self._bar_hits: Dict[tuple, set] = {}
        self.level = -1
        self.avail: List[int] = []
        self.state = "NOT_PLAYED"
        self.cur_key: Optional[int] = None
        self.cur_frame: Optional[np.ndarray] = None
        self.acted_since_reset = False
        self.total_actions = 0
        self._new_level()

    # -- level / graph management -------------------------------------------
    def _new_level(self):
        self.nodes: Dict[int, Node] = {}
        self.root_key: Optional[int] = None
        self.plan: deque = deque()
        self.prev_key: Optional[int] = None
        self.prev_idx: Optional[int] = None          # candidate index at prev node
        self.prev_action: Optional[Tuple[int, int, int]] = None
        self.level_actions = 0
        self.visited_targets = set()
        self.stuck = 0
        if getattr(self, "no_mask", False):
            # masking was switched off for the previous level; start fresh
            self.no_mask = False
            self.vol = getattr(self, "_saved_vol", self.vol)
        self.no_mask = False

    # -- avatar guidance --------------------------------------------------------
    def _targets(self, f: np.ndarray, comps=None):
        """Small/medium non-background objects that are not the avatar."""
        counts = np.bincount(f.ravel().astype(np.int64), minlength=16)
        bg = int(counts.argmax())
        if comps is None:
            comps = components(f, self.vol)
        out = []
        for (c, size, (y, x), bb) in comps:
            if c == bg or c == self.avatar.color or size < 2 or size > 400:
                continue
            out.append((c, y, x))
        return out

    def _tkey(self, c, y, x):
        return (c, y // 4, x // 4)

    def _mark_visited(self, f: np.ndarray):
        p = self.avatar.pos(f, self.vol)
        if p is None:
            return
        py, px, n = p
        r = 2.5 + 0.5 * np.sqrt(n)
        for (c, y, x) in self._targets(f):
            if (y - py) ** 2 + (x - px) ** 2 <= r * r:
                self.visited_targets.add(self._tkey(c, y, x))

    def _key_order(self, f: np.ndarray, comps) -> Dict[int, float]:
        """Order keys so the avatar heads for the nearest unvisited object."""
        p = self.avatar.pos(f, self.vol)
        if p is None:
            return {}
        py, px, _ = p
        best, bd = None, 1e9
        for (c, y, x) in self._targets(f, comps):
            if self._tkey(c, y, x) in self.visited_targets:
                continue
            d = (y - py) ** 2 + (x - px) ** 2
            if d < bd:
                best, bd = (y, x), d
        if best is None or bd < 1:
            return {}
        vy, vx = best[0] - py, best[1] - px
        vn = np.hypot(vy, vx)
        out = {}
        for a in self.avail:
            if a not in SIMPLE_IDS:
                continue
            dv = self.avatar.disp(a)
            if dv is None:
                out[a] = 0.0
                continue
            dn = np.hypot(dv[0], dv[1])
            out[a] = -(dv[0] * vy + dv[1] * vx) / (dn * vn + 1e-9) if dn > 0 else 0.5
        return out

    def _key(self, frame: np.ndarray) -> int:
        f = frame.copy()
        f[self.vol] = 0
        return hash(f.tobytes())

    def _get_node(self, frame: np.ndarray, depth: int) -> Node:
        k = self._key(frame)
        n = self.nodes.get(k)
        if n is None:
            n = Node(k, frame, depth)
            self._build_candidates(n, frame)
            if len(self.nodes) < MAX_NODES:
                self.nodes[k] = n
        return n

    def _rekey(self):
        """Mask changed: recompute keys and merge nodes that now coincide."""
        remap: Dict[int, int] = {}
        new_nodes: Dict[int, Node] = {}
        for old_k, n in self.nodes.items():
            nk = self._key(n.frame)
            remap[old_k] = nk
            tgt = new_nodes.get(nk)
            if tgt is None:
                n.key = nk
                new_nodes[nk] = n
                continue
            for i in range(len(n.acts)):
                if n.status[i] == UNTESTED:
                    continue
                j = tgt.index_of(n.act(i))
                if j is None or tgt.status[j] != UNTESTED:
                    continue
                tgt.status[j] = n.status[i]
                if i in n.edges:
                    tgt.edges[j] = n.edges[i]
                if i in n.dead:
                    tgt.dead[j] = n.dead[i]
            tgt.ptr = 0
            tgt.visits += n.visits
            tgt.depth = min(tgt.depth, n.depth)
        for n in new_nodes.values():
            n.edges = {i: remap.get(d, d) for i, d in n.edges.items()}
            for i, d in list(n.edges.items()):
                if d == n.key:             # a former move is now a self-loop
                    del n.edges[i]
                    n.status[i] = NOOP
        self.nodes = new_nodes
        self.root_key = remap.get(self.root_key, self.root_key)
        self.cur_key = remap.get(self.cur_key, self.cur_key)
        self.prev_key = remap.get(self.prev_key, self.prev_key)
        self.plan = deque()

    # -- candidate generation -----------------------------------------------
    def _simple_tier(self, a: int) -> int:
        t, e = self.simple_stats.get(a, (0, 0))
        if t >= 6 and e == 0:
            return 3
        return 0

    def _remember_win(self):
        """Click signatures on the shortest known root->win path get top
        priority in later levels (levels of a game reuse mechanics)."""
        try:
            if self.root_key is None or self.prev_key is None:
                return
            parent = {self.root_key: None}
            q = deque([self.root_key])
            while q and self.prev_key not in parent:
                k = q.popleft()
                n = self.nodes.get(k)
                if n is None:
                    continue
                for i, nk in n.edges.items():
                    if nk not in parent:
                        parent[nk] = (k, i)
                        q.append(nk)
            steps = []
            if self.prev_key in parent:
                steps = self._reconstruct(parent, self.prev_key)
            if self.prev_idx is not None:
                steps.append((self.prev_key, self.prev_idx))
            for k, i in steps:
                n = self.nodes.get(k)
                if n is not None and i is not None and int(n.acts[i, 0]) == 6:
                    self.win_sigs.add(int(n.sig[i]))
        except Exception:
            pass

    def _click_tier(self, sid: int, base: int) -> int:
        if sid in self.win_sigs:
            return 0
        t, e = self.sig_stats[sid]
        if t >= 1 and e > 0:
            return max(0, base - 1) if e / t > 0.3 else base
        if t >= 3 and e == 0:
            return min(N_TIERS - 1, base + 2)
        if t >= 1 and e == 0:
            return min(N_TIERS - 1, base + 1)
        return base

    def _sig_id(self, sig: tuple) -> int:
        sid = self.sig_ids.get(sig)
        if sid is None:
            sid = len(self.sig_stats)
            self.sig_ids[sig] = sid
            self.sig_stats.append([0, 0])
        return sid

    def _build_candidates(self, node: Node, f: np.ndarray):
        rows = []   # (tier, order, aid, x, y, sid)
        comps = None
        korder: Dict[int, float] = {}
        if self.avatar.color is not None and any(a in SIMPLE_IDS for a in self.avail):
            comps = components(f, self.vol)
            korder = self._key_order(f, comps)
        for a in self.avail:
            if a in SIMPLE_IDS:
                rows.append((self._simple_tier(a), korder.get(a, 0.0) + a * 1e-3, a, -1, -1, -1))
        if 6 in self.avail:
            counts = np.bincount(f.ravel().astype(np.int64), minlength=16)
            bg = int(counts.argmax())
            seen = set()
            if comps is None:
                comps = components(f, self.vol)
            for (c, size, (y, x), bb) in comps:
                if (x, y) in seen:
                    continue
                seen.add((x, y))
                h = bb[2] - bb[0] + 1
                w = bb[3] - bb[1] + 1
                if c == bg and size > 300:
                    base = 4
                elif size > 600:
                    base = 3
                elif size == 1 or size > 150:
                    base = 2
                else:
                    base = 1
                sid = self._sig_id((c, min(size, 999), h, w))
                t = self._click_tier(sid, base)
                rarity = counts[c] / f.size
                rows.append((t, rarity + self.rng.random() * 1e-3, 6, x, y, sid))
        rows.sort(key=lambda z: (z[0], z[1]))
        k = len(rows)
        node.acts = np.array([(r[2], r[3], r[4]) for r in rows], np.int16).reshape(k, 3)
        node.tier = np.array([r[0] for r in rows], np.int8)
        node.sig = np.array([r[5] for r in rows], np.int32)
        node.status = np.zeros(k, np.int8)

    # -- observation ----------------------------------------------------------
    def observe(self, frame, state: str, levels_completed: int, available_actions):
        f = np.asarray(frame, dtype=np.int8)
        if f.ndim == 3:
            f = f[-1]
        if f.shape != (64, 64):
            f = self.cur_frame if self.cur_frame is not None else np.zeros((64, 64), np.int8)
        self.state = state
        if available_actions:
            self.avail = sorted(int(a) for a in available_actions)

        if levels_completed != self.level:
            if self.level >= 0 and levels_completed > self.level:
                self._record_stats(self.prev_key, self.prev_idx, True)
                self._remember_win()
            self.level = levels_completed
            self._new_level()
            self.acted_since_reset = False
            self.cur_frame = f
            root = self._get_node(f, 0)
            self.root_key = root.key
            self.cur_key = root.key
            root.visits += 1
            return

        prev_frame = self.cur_frame
        self.cur_frame = f
        was_reset = self.prev_action is not None and self.prev_action[0] == RESET

        if self.prev_action is not None and prev_frame is not None and not was_reset:
            self._learn_volatile(prev_frame, f)
            if self.prev_action[0] in SIMPLE_IDS and state != "GAME_OVER":
                self.avatar.update(prev_frame, f, self.prev_action[0], self.vol)
        if self.avatar.color is not None and state != "GAME_OVER":
            self._mark_visited(f)

        prev = self.nodes.get(self.prev_key) if self.prev_key is not None else None
        pi = self.prev_idx
        if state == "GAME_OVER":
            if prev is not None and pi is not None:
                prev.dead[pi] = prev.dead.get(pi, 0) + 1
                # A move that was already known to be harmless most likely
                # died from running out of energy/time: keep the edge.
                if prev.status[pi] == UNTESTED or prev.dead[pi] >= 3:
                    prev.status[pi] = DEAD
                    prev.edges.pop(pi, None)
            self.cur_key = None
            self.plan = deque()
            return

        depth = (prev.depth + 1) if prev is not None else 0
        node = self._get_node(f, depth)
        node.visits += 1
        if node.apos is None and self.avatar.color is not None:
            p = self.avatar.pos(f, self.vol)
            if p is not None:
                node.apos = (p[0], p[1])
        if self.root_key is None or (was_reset and self.no_mask):
            self.root_key = node.key

        if prev is not None and pi is not None and not was_reset:
            first = prev.status[pi] == UNTESTED
            if node.key == prev.key:
                if first:
                    self._record_stats(prev.key, pi, False)
                prev.status[pi] = NOOP
                prev.edges.pop(pi, None)
            else:
                if first:
                    self._record_stats(prev.key, pi, True)
                prev.status[pi] = MOVED
                prev.edges[pi] = node.key
        self.cur_key = node.key

    @staticmethod
    def _comp_at(frame: np.ndarray, y: int, x: int):
        m = frame == frame[y, x]
        lab, _ = _label(m)
        c = lab == lab[y, x]
        ys, xs = np.nonzero(c)
        return (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())), len(ys)

    def _learn_volatile(self, a: np.ndarray, b: np.ndarray):
        """Detect step counters / energy bars and exclude them from state keys.

        Cues: (1) tiny changes inside the outer border band; (2) a lone small
        change (nothing else moved) at the end of a thin, long bar, seen for
        at least two different actions (a timer ticking)."""
        if self.no_mask:
            return
        d = (a != b) & ~self.vol
        if not d.any():
            return
        band = np.zeros((64, 64), bool)
        band[:BORDER, :] = band[-BORDER:, :] = True
        band[:, :BORDER] = band[:, -BORDER:] = True
        lab, n = _label(d)
        if n > 12:
            return
        changed = False
        for i in range(1, n + 1):
            comp = lab == i
            size = int(comp.sum())
            if size > 6:
                continue
            ys, xs = np.nonzero(comp)
            if size <= 4 and not (comp & ~band).any():
                for y, x in zip(ys, xs):
                    if y < BORDER or y >= 64 - BORDER:
                        if not self.vol[y, :].all():
                            self.vol[y, :] = True
                            changed = True
                    elif x < BORDER or x >= 64 - BORDER:
                        if not self.vol[:, x].all():
                            self.vol[:, x] = True
                            changed = True
                continue
            if n != 1 or size > 4:
                continue
            y, x = int(ys[0]), int(xs[0])
            for fr in (a, b):
                col = int(fr[y, x])
                if col == int(np.bincount(fr.ravel().astype(np.int64), minlength=16).argmax()):
                    continue
                (y0, x0, y1, x1), sz = self._comp_at(fr, y, x)
                h, w = y1 - y0 + 1, x1 - x0 + 1
                if min(h, w) <= 3 and max(h, w) >= 8 and sz >= max(h, w):
                    bkey = (col, y0 if w >= h else x0, w >= h)
                    acts = self._bar_hits.setdefault(bkey, set())
                    acts.add(self.prev_action)
                    if len(acts) < 2:
                        break
                    y0m, y1m = max(0, min(y0, int(ys.min()))), min(63, max(y1, int(ys.max())))
                    x0m, x1m = max(0, min(x0, int(xs.min()))), min(63, max(x1, int(xs.max())))
                    if w >= h:
                        x0m, x1m = 0, 63
                    else:
                        y0m, y1m = 0, 63
                    reg = self.vol[y0m:y1m + 1, x0m:x1m + 1]
                    if self.vol.sum() + reg.size > 0.2 * self.vol.size:
                        break
                    if not reg.all():
                        reg[...] = True
                        changed = True
                    break
        if changed:
            self._rekey()

    def _record_stats(self, key: Optional[int], idx: Optional[int], effective: bool):
        if key is None or idx is None:
            return
        node = self.nodes.get(key)
        if node is None or idx >= len(node.acts):
            return
        aid = int(node.acts[idx, 0])
        if aid == 6:
            s = self.sig_stats[int(node.sig[idx])]
        else:
            s = self.simple_stats.setdefault(aid, [0, 0])
        s[0] += 1
        if effective:
            s[1] += 1

    # -- decision -------------------------------------------------------------
    def next_action(self) -> Tuple[int, Optional[dict]]:
        key, idx = self._decide()
        node = self.nodes.get(key) if key is not None else None
        self.total_actions += 1
        self.level_actions += 1
        if idx is None or node is None:
            self.prev_key = self.cur_key
            self.prev_idx = None
            self.prev_action = (RESET, -1, -1)
            self.acted_since_reset = False
            self.cur_key = None
            return RESET, None
        act = node.act(idx)
        self.prev_key = key
        self.prev_idx = idx
        self.prev_action = act
        self.acted_since_reset = True
        if act[0] == 6:
            return 6, {"x": act[1], "y": act[2]}
        return act[0], None

    def _decide(self) -> Tuple[Optional[int], Optional[int]]:
        """Returns (node_key, candidate_idx); idx None means RESET."""
        if self.state in ("NOT_PLAYED", "GAME_OVER") or self.cur_key is None:
            return None, None
        node = self.nodes.get(self.cur_key)
        if node is None:
            return None, None

        if self.plan:
            exp_key, i = self.plan[0]
            if exp_key == node.key:
                self.plan.popleft()
                if i is None and not self.acted_since_reset:
                    self.plan = deque()
                else:
                    return node.key, i
            else:
                self.plan = deque()

        best = node.best_untested()
        if best is not None and node.tier[best] == 0:
            return node.key, best

        path = self._plan_to_frontier(node, best)
        if path is None:
            if best is not None:
                return node.key, best
            return self._fallback(node)
        if not path:
            return node.key, best
        self.plan = deque(path)
        exp_key, i = self.plan.popleft()
        return exp_key, i

    def _plan_to_frontier(self, node: Node, best: Optional[int]):
        """BFS over known edges (plus RESET->root). Returns a list of
        (node_key, cand_idx or None for RESET) to reach the frontier node whose
        best untested candidate is cheapest; [] means act here.
        With a known avatar, frontier nodes where it stands close to an
        unvisited object are preferred (distance in avatar steps)."""
        hfun = self._target_heuristic()
        h0 = hfun(node) if hfun else 0.0
        best_cost = (float(node.tier[best]) * TIER_COST + h0) if best is not None else float("inf")
        best_path: Optional[list] = [] if best is not None else None

        dist = {node.key: 0}
        parent: Dict[int, Optional[Tuple[int, Optional[int]]]] = {node.key: None}
        q = deque([node.key])
        if (self.root_key is not None and self.root_key != node.key
                and self.acted_since_reset and self.root_key in self.nodes):
            dist[self.root_key] = 1
            parent[self.root_key] = (node.key, None)
            q.append(self.root_key)
        budget = 30000
        nodes = self.nodes
        while q and budget > 0:
            k = q.popleft()
            budget -= 1
            d = dist[k]
            if d >= best_cost:
                continue
            n = nodes.get(k)
            if n is None:
                continue
            if k != node.key:
                b = n.best_untested()
                if b is not None:
                    c = d + float(n.tier[b]) * TIER_COST
                    if hfun and c < best_cost:
                        c += hfun(n)
                    if c < best_cost:
                        best_cost = c
                        best_path = self._reconstruct(parent, k)
            for i, nk in n.edges.items():
                if nk not in dist and nk in nodes:
                    dist[nk] = d + 1
                    parent[nk] = (k, i)
                    q.append(nk)
        return best_path

    def _target_heuristic(self):
        if self.avatar.color is None or self.cur_frame is None or HEUR_W <= 0:
            return None
        ts = [(y, x) for (c, y, x) in self._targets(self.cur_frame)
              if self._tkey(c, y, x) not in self.visited_targets]
        if not ts:
            return None
        steps = []
        for a in self.avail:
            if a in SIMPLE_IDS:
                dv = self.avatar.disp(a)
                if dv is not None:
                    steps.append(np.hypot(dv[0], dv[1]))
        step = max(1.0, float(np.median(steps))) if steps else 1.0
        T = np.array(ts, np.float64)

        def h(n: Node) -> float:
            if n.apos is None:
                return 0.0
            d2 = (T[:, 0] - n.apos[0]) ** 2 + (T[:, 1] - n.apos[1]) ** 2
            return HEUR_W * float(np.sqrt(d2.min())) / step
        return h

    @staticmethod
    def _reconstruct(parent, k):
        path = []
        while parent[k] is not None:
            pk, i = parent[k]
            path.append((pk, i))
            k = pk
        path.reverse()
        return path

    def _fallback(self, node: Node) -> Tuple[Optional[int], Optional[int]]:
        # Everything reachable is exhausted: retry the least-tried dead end,
        # otherwise a random candidate. If this keeps happening the mask
        # probably hides game state: drop it for the rest of the level.
        self.stuck += 1
        if self.stuck == 150 and self.vol.any() and not self.no_mask:
            self._saved_vol = self.vol.copy()
            self.no_mask = True
            self.vol = np.zeros((64, 64), bool)
            self.nodes = {}
            self.root_key = None
            self.plan = deque()
            self.prev_key = None
            self.prev_idx = None
            if self.cur_frame is not None:
                n = self._get_node(self.cur_frame, 0)
                self.cur_key = n.key
                if n.best_untested() is not None:
                    return n.key, n.best_untested()
        if node.dead:
            i = min(node.dead, key=lambda z: node.dead[z])
            return node.key, i
        if len(node.acts):
            return node.key, self.rng.randrange(len(node.acts))
        if self.acted_since_reset:
            return None, None
        return node.key, None if not len(node.acts) else 0
