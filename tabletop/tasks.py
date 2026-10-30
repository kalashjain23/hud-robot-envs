"""Tabletop tasks. ``env`` is re-exported so ``hud eval tabletop/tasks.py`` serves it."""

from __future__ import annotations

import sys
from pathlib import Path

from hud import Taskset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tabletop.env import env, lift, place, place_both  # noqa: F401

_lift = lift(seed=0)
_lift.slug = "lift-cube"
_place = place(seed=1)
_place.slug = "place-cube"
_place_both = place_both(seed=2)
_place_both.slug = "place-both-cubes"

taskset = Taskset("tabletop", [_lift, _place, _place_both])
