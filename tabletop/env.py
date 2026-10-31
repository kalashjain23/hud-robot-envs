"""The Franka tabletop: the MuJoCo sim behind a ``robot`` wire, plus a control interface for LLMs.

``TABLETOP_INTERFACE`` picks how an LLM drives it: ``direct`` (one tool call per motion),
``code`` (programs over motion primitives, one ``run`` call each) or ``supervise`` (SmolVLA
fine-tuned on LIBERO acts, zero-shot, from the local server in
``libero_env/smolvla_server.py``; the LLM instructs it and can take over with the motion tool;
needs ``TABLETOP_CONTROL=delta``).

``TABLETOP_CONTROL`` picks the level the arm is driven at (``ee``, ``joint`` or ``delta``),
which picks the contract and so the direct-control tool (``move_to``, ``move_joints``,
``move_by``). ``TABLETOP_VIDEO`` names a directory for one MP4 per graded episode::

    TABLETOP_CONTROL=joint TABLETOP_VIDEO=runs hud eval tabletop/tasks.py claude --max-steps 20
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from hud import Environment
from hud.agents.robot import RemoteModel
from hud.environment.robot import CodeAsPolicy, DirectControl, PolicySupervision, RobotEndpoint

from tabletop.policy import LiberoPolicyAdapter

CONTROL = os.environ.get("TABLETOP_CONTROL", "ee")
INTERFACE = os.environ.get("TABLETOP_INTERFACE", "direct")
if INTERFACE not in ("direct", "code", "supervise"):
    raise ValueError(f"TABLETOP_INTERFACE must be direct, code or supervise, got {INTERFACE!r}")
if INTERFACE == "supervise" and CONTROL != "delta":
    raise ValueError("TABLETOP_INTERFACE=supervise needs TABLETOP_CONTROL=delta")
PORT = int(os.environ.get("TABLETOP_POLICY_PORT", "8000"))
#: Where the sim writes one MP4 per graded episode (unset: no video).
VIDEO = os.environ.get("TABLETOP_VIDEO", "")
#: How camera mounts are published: live (default), static, or off (to benchmark them).
GEOMETRY = os.environ.get("TABLETOP_GEOMETRY", "live")
if GEOMETRY not in ("live", "static", "off"):
    raise ValueError(f"TABLETOP_GEOMETRY must be live, static or off, got {GEOMETRY!r}")
#: Camera image size in pixels (square).
IMAGE = int(os.environ.get("TABLETOP_IMAGE", "256"))

SCENE = (
    "World frame in metres: the robot base is at the origin, the table top is z = 0, +x points "
    "away from the base and +y to its left. The light shines straight down, "
    "so shadows sit directly under objects. The cubes are 4 cm wide; the plates are 11 cm "
    "across and 1 cm tall. Grip: -1 is fully open and +1 fully closed."
)
if GEOMETRY == "static":
    # A static wrist mount is relative to the hand, which only notes can describe.
    SCENE += (
        " The hand frame turns with the fingertip: its +z points out along the fingers, the "
        "fingers close along its y, and its origin is 10 cm back from the fingertip."
    )
NOTES = {
    "ee": (
        "Targets are the fingertip centre (between the finger pads). Pointing straight down is "
        "roll = pi (same as -pi), pitch = 0; yaw turns the hand about the vertical, and at "
        "yaw = 0 the fingers close along y. The arm sags a few mm under load."
    ),
    "joint": (
        "Targets are the seven arm joint angles in radians, base (joint1) to wrist (joint7), "
        "each servoed directly; there is no inverse kinematics. The fingertip pose is reported "
        "under observation/eef for reference."
    ),
    "delta": (
        "Displacements move the fingertip centre in the world frame, in metres; the hand keeps "
        "pointing straight down. The grip is a displacement too: +2 closes a fully open hand."
    ),
}

env = Environment(name="tabletop")
sim = RobotEndpoint.spawn(
    [
        sys.executable,
        "-m",
        "hud.environment.robot.bridge",
        f"{Path(__file__).with_name('sim.py')}:Franka",
        "--init",
        json.dumps(
            {
                "control": CONTROL,
                "video": VIDEO,
                "geometry": GEOMETRY,
                "image": IMAGE,
            }
        ),
    ]
)
sim.attach(env)
options: dict[str, Any] = {
    "notes": f"{NOTES[CONTROL]} {SCENE}",
    "max_step": {"gripper": 0.5},
    # The joint servos settle within a few mm of a pose; closer is not "stopped short".
    "tolerance": {"position": 0.008},
    "time_limit": 120,
}
if INTERFACE == "supervise":
    # 50 LIBERO steps per SmolVLA chunk are 25 tabletop ticks, all played as LeRobot plays them.
    adapter = LiberoPolicyAdapter(chunk_size=25)
    PolicySupervision(RemoteModel(port=PORT), adapter, use_motion=True, **options).attach(sim)
else:
    (DirectControl if INTERFACE == "direct" else CodeAsPolicy)(**options).attach(sim)


@env.initialize
async def up() -> None:
    await sim.start()
    for capability in await sim.capabilities():
        env.add_capability(capability)


@env.shutdown
async def down() -> None:
    await sim.stop()


@env.template()
async def lift(seed: int = 0):
    ep = await sim.reset(task="lift", seed=seed)
    yield {"prompt": ep["prompt"]}
    yield await sim.result()


@env.template()
async def place(seed: int = 0):
    ep = await sim.reset(task="place", seed=seed)
    yield {"prompt": ep["prompt"]}
    yield await sim.result()


@env.template()
async def place_both(seed: int = 0):
    ep = await sim.reset(task="place_both", seed=seed)
    yield {"prompt": ep["prompt"]}
    yield await sim.result()
