"""LIBERO as a HUD env: the benchmark's sim behind a ``robot`` wire, plus a control interface
for LLMs.

The sim runs in LIBERO's own venv (``third_party/libero-venv``, see the README);
this module runs in the repo's. ``LIBERO_INTERFACE`` picks how an LLM drives it: ``direct``
(one tool call per motion) or ``code`` (programs over motion primitives). The action is
LIBERO's end-effector delta, so the motion tool is ``move_by``. ``LIBERO_VIDEO`` names a
directory for one MP4 per graded episode.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from hud import Environment
from hud.environment.robot import CodeAsPolicy, DirectControl, RobotEndpoint

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "third_party" / "libero-venv" / "bin" / "python"
SUITE = os.environ.get("LIBERO_SUITE", "libero_10")
INTERFACE = os.environ.get("LIBERO_INTERFACE", "direct")
if INTERFACE not in ("direct", "code"):
    raise ValueError(f"LIBERO_INTERFACE must be direct or code, got {INTERFACE!r}")
VIDEO = os.environ.get("LIBERO_VIDEO", "")
#: openpi's LIBERO eval budget per suite, in 20 Hz steps.
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}

# Datasheet facts only: the controller's scaling and the gripper's behaviour (robosuite's
# OSC_POSE and PandaGripper as LIBERO configures them), and the state's units.
NOTES = (
    "Each step's arm action is a displacement target for an operational-space controller in "
    "the world frame: x, y, z of +-1 command +-5 cm, and roll, pitch, yaw of +-1 command "
    "+-0.5 rad about the world x, y, z axes, scaled linearly. The arm tracks the target at "
    "20 Hz and moves less than commanded within a step. The gripper is a direction, not a "
    "position: grip +1 closes, -1 opens, 0 holds, and the fingers take about 8 steps to go "
    "from open to closed. State: eef.x, eef.y, eef.z are the point between the fingertips in "
    "metres in the world frame, eef.rx, eef.ry, eef.rz its orientation as axis-angle, and "
    "finger.left, finger.right each finger's opening (about 0.04 m open, 0 closed)."
)

env = Environment(name="libero")
sim = RobotEndpoint.spawn(
    [
        str(PYTHON),
        "-m",
        "hud.environment.robot.bridge",
        f"{Path(__file__).with_name('sim.py')}:Libero",
        "--init",
        json.dumps({"suite": SUITE, "video": VIDEO}),
    ]
)
sim.attach(env)
interface = DirectControl if INTERFACE == "direct" else CodeAsPolicy
interface(notes=NOTES, time_limit=MAX_STEPS[SUITE] / 20).attach(sim)


@env.initialize
async def up() -> None:
    await sim.start()
    for capability in await sim.capabilities():
        env.add_capability(capability)


@env.shutdown
async def down() -> None:
    await sim.stop()


@env.template()
async def libero(task_id: int = 0, init_state: int = 0):
    ep = await sim.reset(task_id=task_id, init_state=init_state)
    yield {"prompt": ep["prompt"]}
    yield await sim.result()
