"""LIBERO-Goal tasks. ``env`` is re-exported so ``hud eval libero_env/tasks.py`` serves it.

The long-horizon chains are in ``goal_chains.py``, which sets ``LIBERO_TASKS_PER_EPISODE=5``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from hud import Taskset

os.environ.setdefault("LIBERO_SUITE", "libero_goal")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from libero_env.env import env, libero  # noqa: F401

_drawer = libero(task_id=0, init_state=0)
_drawer.slug = "libero-goal-open-middle-drawer"
_bowl = libero(task_id=8, init_state=0)
_bowl.slug = "libero-goal-bowl-on-plate"

taskset = Taskset("libero-goal", [_drawer, _bowl])
