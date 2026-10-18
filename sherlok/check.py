"""Compare what the camera sees with the schema, at the level of electrical nodes (strips).

Schema JSON:
  {"name": "LED demo",
   "parts": [{"id": "R1", "type": "resistor", "pins": ["a5", "a10"]},
             {"id": "W1", "type": "wire",     "pins": ["j10", "tn3"]}]}

Holes in the same strip are equivalent (a5 == c5), so pins are compared by node.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .board import Board
from .vision import Frame

GREEN, YELLOW, RED = "green", "yellow", "red"


@dataclass
class Result:
    state: str                       # instantaneous state for this frame
    issues: list = field(default_factory=list)   # (severity, message)
    matched: dict = field(default_factory=dict)  # part id -> detected part index
    extra: list = field(default_factory=list)    # detected part indices not in schema


def load_schema(path) -> dict:
    """Load a schema. Two formats:
    .txt  one part per line:  ID TYPE PIN PIN      e.g.  R1 resistor c5 c9    (# = comment)
    .json {"name": ..., "parts": [{"id", "type", "pins": [...]}, ...]}
    """
    path = Path(path)
    if path.suffix.lower() == ".txt":
        parts = []
        for ln in path.read_text().splitlines():
            ln = ln.split("#")[0].split()
            if not ln:
                continue
            if len(ln) < 4:
                raise ValueError(f"{path}: need 'ID TYPE PIN PIN', got {' '.join(ln)!r}")
            parts.append({"id": ln[0], "type": ln[1].lower(), "pins": ln[2:]})
        s = {"name": path.stem, "parts": parts}
    else:
        s = json.loads(path.read_text())
    for p in s["parts"]:
        p["pins"] = [x.strip().lower() for x in p["pins"]]
    return s


def find_schema(folder, name: str) -> Path | None:
    for ext in (".txt", ".json"):
        p = Path(folder) / f"{name}{ext}"
        if p.exists():
            return p
    return None


class Checker:
    def __init__(self, board: Board, schema: dict, high_conf: float = 0.8):
        self.b = board
        self.schema = schema
        self.high = high_conf
        rs = board.cfg.get("rails_split", {})
        self.expected = []
        for p in schema["parts"]:
            nodes = frozenset(Board.node_of(x, rs) for x in p["pins"])
            self.expected.append((p, nodes))

    def check(self, f: Frame) -> Result:
        if f.obstructed:
            return Result(YELLOW, [("info", "board obstructed (hand?) - can't verify")])
        det = []
        for i, part in enumerate(f.parts):
            n0 = self.b.holes[part.ends[0]].node
            n1 = self.b.holes[part.ends[1]].node
            det.append((i, frozenset((n0, n1)), part.conf))

        res = Result(GREEN)
        used = set()
        pending = []
        for p, nodes in self.expected:  # pass 1: exact node matches
            hit = next((d for d in det if d[0] not in used and d[1] == nodes), None)
            if hit:
                used.add(hit[0])
                res.matched[p["id"]] = hit[0]
            else:
                pending.append((p, nodes))
        for p, nodes in pending:  # pass 2: one end right, other end wrong -> misplaced
            near = next((d for d in det if d[0] not in used and d[1] & nodes), None)
            if near:
                used.add(near[0])
                got = "-".join(sorted(near[1]))
                want = "-".join(sorted(nodes))
                sev = "error" if near[2] >= self.high else "warn"
                res.issues.append((sev, f"{p['id']} misplaced: expected {want}, found {got}"))
            else:
                res.issues.append(("error", f"{p['id']} ({p.get('type', 'part')}) missing at {'/'.join(p['pins'])}"))
        for i, nodes, conf in det:
            if i in used:
                continue
            res.extra.append(i)
            what = "short (both ends in one strip)" if len(nodes) == 1 else "unexpected connection"
            sev = "error" if conf >= self.high else "warn"
            res.issues.append((sev, f"{what}: {'-'.join(sorted(nodes))}"))

        sevs = {s for s, _ in res.issues}
        res.state = RED if "error" in sevs else YELLOW if "warn" in sevs else GREEN
        return res


class Debouncer:
    """Red only if errors persist `hold` seconds; green only if clean for `hold_green` seconds."""

    def __init__(self, hold: float = 2.0, hold_green: float = 1.0):
        self.hold, self.hold_green = hold, hold_green
        self.hist: deque = deque()
        self.state = YELLOW

    def update(self, state: str, now: float | None = None) -> str:
        now = time.monotonic() if now is None else now
        self.hist.append((now, state))
        while self.hist and now - self.hist[0][0] > max(self.hold, self.hold_green) + 1:
            self.hist.popleft()
        def held(s, secs):
            win = [st for t, st in self.hist if now - t <= secs]
            span = now - min(t for t, _ in self.hist)
            return span >= secs * 0.9 and win and all(st == s for st in win)
        if held(RED, self.hold):
            self.state = RED
        elif held(GREEN, self.hold_green):
            self.state = GREEN
        elif state == YELLOW or self.state == GREEN and state == RED:
            self.state = YELLOW
        return self.state
