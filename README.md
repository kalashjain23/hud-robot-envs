# hud-robot-interfaces

Robot environments and tasks for HUD's robot control interfaces. Each env serves its sim on a
`robot` wire and attaches a control interface, so an LLM drives it through MCP tools:

| Interface | Selected by | What the model gets |
|-----------|-------------|---------------------|
| Direct control | `*_INTERFACE=direct` | One motion tool, derived from the contract (`move_to`, `move_by`, ...), plus `wait` |
| Code as Policy | `*_INTERFACE=code` | One `run` tool executing Python over the same motion primitives |
| Policy Supervision | `*_INTERFACE=supervise` | `run_policy(instruction)`: a LIBERO-trained VLA acts for 2 s on the instruction the model chooses (`LIBERO_POLICY`: `pi05` on Dreamscale, or `smolvla` locally; the tabletop runs SmolVLA zero-shot). The model can take over with the motion tool and hand back with `run_policy`; LIBERO adds `home`, which sends the arm back to its start pose. |

All three interfaces live in the SDK (`hud.environment.robot`); this repo only declares envs
and tasks.

## Environments

| Env | Robot and sim | Action | Cameras |
|-----|---------------|--------|---------|
| `tabletop/` | Franka Panda (MuJoCo Menagerie), two cubes and two plates | end-effector pose (`ee`), joint angles (`joint`) or end-effector deltas (`delta`), via `TABLETOP_CONTROL` | corner camera and a wrist camera, 256 px (`TABLETOP_IMAGE`) |
| `libero_env/` | LIBERO as the benchmark runs it: robosuite Panda, OSC_POSE at 20 Hz | LIBERO's 7-dim end-effector delta | `agentview` and `robot0_eye_in_hand`, 256 px |

Both contracts publish each camera's mount, and a camera on the arm as a live pose.

## Tasks

| Task | Env | Prompt |
|------|-----|--------|
| `lift` | tabletop | Pick up the named cube and lift it 10 cm. |
| `place` | tabletop | Put the named cube on the named plate; the other pair is a distractor. |
| `place_both` | tabletop | Put each named cube on its named plate. Scored 0.5 per pair. |
| `libero` | libero_env | Any task of a LIBERO suite (`task_id`) from one of its fixed init states (`init_state`), graded by LIBERO's own success check. `libero_env/tasks.py` has two from LIBERO-Goal: open the middle drawer of the cabinet, and put the bowl on the plate. |
| `libero_chain` | libero_env | Several LIBERO-Goal tasks in one episode, in order (`libero_env/goal_chains.py` has three five-task chains). Scored by the share of their goals that hold at the end. |

Tabletop layouts and pairings are seeded, and the seed picks the episode.

## Run

```bash
uv sync
TABLETOP_INTERFACE=code uv run hud eval tabletop/tasks.py claude --max-steps 30
LIBERO_INTERFACE=direct uv run hud eval libero_env/tasks.py claude --max-steps 30
LIBERO_INTERFACE=supervise LIBERO_POLICY=smolvla uv run hud eval libero_env/goal_chains.py claude
```

`TABLETOP_VIDEO` / `LIBERO_VIDEO` name a directory for one MP4 per graded episode. With `HUD_API_KEY` set, every episode's trace reaches HUD.

## LIBERO setup

LIBERO needs robosuite 1.4.1, which works only with MuJoCo 3.2.x. The tabletop uses a newer
MuJoCo, so LIBERO's sim runs in its own venv and `libero_env/env.py` spawns the bridge with that
Python:

```bash
cd third_party
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git
uv venv --python 3.11 libero-venv
VIRTUAL_ENV=$PWD/libero-venv uv pip install robosuite==1.4.1 mujoco==3.2.7 bddl==1.0.1 \
    future gym==0.25.2 easydict termcolor pyyaml tqdm imageio matplotlib torch scipy \
    "numpy<2" -e "../../hud-python[robot]"
echo "$PWD/LIBERO" > libero-venv/lib/python3.11/site-packages/libero_src.pth
```

MuJoCo 3.17 breaks robosuite 1.4.1, and 2.3.7 can't load OpenGL on current macOS. The sim
writes LIBERO's path config itself, pointing at the clone.

`LIBERO_POLICY=pi05` (the default) needs a Dreamscale account (`dreamscale login`). Each
episode opens one session, billed by the hour while it is open.

`LIBERO_POLICY=smolvla` reaches `lerobot/smolvla_libero` through a local openpi-protocol server.
LeRobot needs numpy 2 and `hud[robot]` pins numpy below 2, so it has its own venv:

```bash
cd third_party && uv venv --python 3.12 smolvla-venv
VIRTUAL_ENV=$PWD/smolvla-venv uv pip install "lerobot[smolvla]" msgpack websockets
VIRTUAL_ENV=$PWD/smolvla-venv uv pip install --no-deps openpi-client
cd .. && third_party/smolvla-venv/bin/python libero_env/smolvla_server.py --port 8000
```

The tabletop's `supervise` uses the same server (`TABLETOP_POLICY_PORT`) and needs
`TABLETOP_CONTROL=delta`.
