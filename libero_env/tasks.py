"""LIBERO tasks. ``env`` is re-exported so ``hud eval libero_env/tasks.py`` serves it."""

from __future__ import annotations

import sys
from pathlib import Path

from hud import Taskset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from libero_env.env import env, libero  # noqa: F401

_soup_and_sauce = libero(task_id=0, init_state=0)
_soup_and_sauce.slug = "libero10-soup-and-sauce-in-basket"

taskset = Taskset("libero", [_soup_and_sauce])
