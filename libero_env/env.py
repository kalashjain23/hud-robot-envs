"""LIBERO as a HUD env: the benchmark's sim behind a ``robot`` wire, plus a control interface
for LLMs.

The sim runs in LIBERO's own venv (``third_party/libero-venv``, see the README);
this module runs in the repo's. ``LIBERO_INTERFACE`` picks how an LLM drives it: ``direct``
(one tool call per motion), ``code`` (programs over motion primitives) or ``supervise``
(a VLA fine-tuned on LIBERO acts 2 s per call; the LLM instructs it, can send the arm
home, ``libero_env/home.py``, and can take over with the motion tool, then hand back).
``LIBERO_POLICY`` picks the VLA: ``pi05`` (Dreamscale) or ``smolvla`` (the local server in
``libero_env/smolvla_server.py``, on ``LIBERO_POLICY_PORT``). The action is LIBERO's
end-effector delta, so the motion tool is ``move_by``. ``LIBERO_VIDEO`` names a directory for
one MP4 per graded episode.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from hud import Environment
from hud.agents.robot import RemoteModel
from hud.environment.robot import CodeAsPolicy, DirectControl, PolicySupervision, RobotEndpoint

from libero_env.home import home
from libero_env.policy import DreamscaleModel, Pi05LiberoAdapter, SmolVLALiberoAdapter

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "third_party" / "libero-venv" / "bin" / "python"
SUITE = os.environ.get("LIBERO_SUITE", "libero_10")
INTERFACE = os.environ.get("LIBERO_INTERFACE", "direct")
if INTERFACE not in ("direct", "code", "supervise"):
    raise ValueError(f"LIBERO_INTERFACE must be direct, code or supervise, got {INTERFACE!r}")
POLICY = os.environ.get("LIBERO_POLICY", "pi05")
if POLICY not in ("pi05", "smolvla"):
    raise ValueError(f"LIBERO_POLICY must be pi05 or smolvla, got {POLICY!r}")
PORT = int(os.environ.get("LIBERO_POLICY_PORT", "8000"))
VIDEO = os.environ.get("LIBERO_VIDEO", "")
#: Camera frames' size in px. The policies take 256 (a larger render is averaged down for
#: them), so a larger one only changes what the LLM sees.
IMAGE = int(os.environ.get("LIBERO_IMAGE", "256"))
#: The suite's task instructions, as LIBERO words them (from its BDDL file names): what the
#: policies were trained on.
INSTRUCTIONS = sorted(
    path.stem.replace("_", " ")
    for path in (ROOT / "third_party/LIBERO/libero/libero/bddl_files" / SUITE).glob("*.bddl")
)
#: Tasks per episode: 1, or a chain's length (``libero_chain``); the time limit scales with it.
TASKS_PER_EPISODE = int(os.environ.get("LIBERO_TASKS_PER_EPISODE", "1"))
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
        json.dumps({"suite": SUITE, "video": VIDEO, "image": IMAGE}),
    ]
)
sim.attach(env)
dreamscale = DreamscaleModel()
#: Sim seconds per task (default: openpi's budget for the suite).
TASK_SECONDS = float(os.environ.get("LIBERO_TASK_SECONDS", MAX_STEPS[SUITE] / 20))
time_limit = TASK_SECONDS * TASKS_PER_EPISODE
if INTERFACE == "supervise":
    # Actions played per chunk before the policy is queried again, as each is evaluated:
    # openpi replans pi0.5 every 5, LeRobot plays SmolVLA's whole 50-action chunk.
    if POLICY == "pi05":
        policy, adapter = dreamscale, Pi05LiberoAdapter(chunk_size=5)
    else:
        policy, adapter = RemoteModel(port=PORT), SmolVLALiberoAdapter(chunk_size=50)
    PolicySupervision(
        policy,
        adapter,
        use_motion=True,
        home=home,
        notes=NOTES,
        instructions=INSTRUCTIONS,
        time_limit=time_limit,
    ).attach(sim)
else:
    interface = DirectControl if INTERFACE == "direct" else CodeAsPolicy
    interface(notes=NOTES, time_limit=time_limit).attach(sim)


@env.initialize
async def up() -> None:
    await sim.start()
    for capability in await sim.capabilities():
        env.add_capability(capability)


@env.shutdown
async def down() -> None:
    dreamscale.close()
    await sim.stop()


@env.template()
async def libero(task_id: int = 0, init_state: int = 0):
    ep = await sim.reset(task_id=task_id, init_state=init_state)
    yield {"prompt": ep["prompt"]}
    yield await sim.result()


@env.template()
async def libero_chain(chain: list[int], init_state: int = 0):
    """Several of the suite's tasks in one episode, in order (they must share one scene, as
    ``libero_goal``'s do); scored by the share of their goals that hold at the end."""
    if len(chain) != TASKS_PER_EPISODE:
        raise ValueError(f"a {len(chain)}-task chain needs LIBERO_TASKS_PER_EPISODE={len(chain)}")
    ep = await sim.reset(chain=chain, init_state=init_state)
    yield {"prompt": ep["prompt"]}
    yield await sim.result()
