"""Five-task chains in LIBERO-Goal's one scene: the long-horizon comparison of a policy given
the whole chain in one prompt against an agent supervising it. Each chain's tasks can all hold
at once (the bowl and the wine bottle go to one place each), and they are ordered so no task
leaves something in the way of a later one: the stove is turned on before anything is put in
front of it, the plate is pushed before the bowl goes on it, the cream cheese goes in the bowl
once the bowl is in place, and an open drawer, which sticks out into the workspace, comes last.

    LIBERO_INTERFACE=supervise uv run hud eval libero_env/goal_chains.py claude
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from hud import Taskset

os.environ.setdefault("LIBERO_SUITE", "libero_goal")
os.environ.setdefault("LIBERO_TASKS_PER_EPISODE", "5")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from libero_env.env import env, libero_chain  # noqa: F401

#: Task ids in ``libero_goal`` order.
CHAINS = {
    "stove-plate-bowl-cheese-rack": [7, 5, 8, 6, 9],
    "stove-rack-plate-bowl-drawer": [7, 9, 5, 8, 0],
    "stove-cabinet-bowlstove-cheese-plate": [7, 2, 1, 6, 5],
}

tasks = []
for name, chain in CHAINS.items():
    task = libero_chain(chain=chain, init_state=0)
    task.slug = f"libero-goal-chain-{name}"
    tasks.append(task)

taskset = Taskset("libero-goal-chains", tasks)
