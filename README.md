# hud-robot-interfaces

Robot envs and tasks for HUD's three robot control interfaces. The interfaces live in the SDK
(`hud.environment.robot`); this repo only declares envs and tasks.

| Interface | `*_INTERFACE` | What the LLM gets |
|-----------|---------------|-------------------|
| Direct control | `direct` | One motion tool from the contract (`move_to`, `move_by`, ...) and `wait` |
| Code as Policy | `code` | A `run` tool executing Python over the same motion primitives |
| Policy Supervision | `supervise` | A VLA acting in parallel: `run_policy`, `stop_policy`, `wait`, and optionally `detect`, the motion tool, `home` and `reach` |

## Run

```bash
uv sync
TABLETOP_INTERFACE=code uv run hud eval tabletop/tasks.py claude --max-steps 30
LIBERO_INTERFACE=direct uv run hud eval libero_env/tasks.py claude --max-steps 30
LIBERO_INTERFACE=supervise LIBERO_POLICY=smolvla uv run hud eval libero_env/goal_chains.py claude
```

LIBERO first needs its own venv, and `smolvla` and `detect` need local servers. See [Setup](#setup).
With `HUD_API_KEY` set, every episode's trace reaches HUD.

## Envs and tasks

| Env | Sim | Action | Tasks |
|-----|-----|--------|-------|
| `tabletop/` | Franka Panda (MuJoCo Menagerie), two cubes, two plates | `TABLETOP_CONTROL`: `ee` pose, `joint` angles or `delta` | `lift`, `place`, `place_both` (seeded layouts) |
| `libero_env/` | LIBERO's robosuite Panda, OSC_POSE at 20 Hz | 7-dim end-effector delta | `libero` (one task of a suite, graded by LIBERO), `libero_chain` (several tasks in one episode, scored by the share that hold) |

`tabletop/tasks.py`, `libero_env/tasks.py` (open the middle drawer, put the bowl on the plate) and
`libero_env/goal_chains.py` (three five-task chains) hold the tasks. Both envs publish each
camera's pose, so the LLM can turn pixels into world points.

## Options

Environment variables, `TABLETOP_` or `LIBERO_` prefixed:

| Variable | Default | Effect |
|----------|---------|--------|
| `INTERFACE` | `direct` | `direct`, `code` or `supervise` |
| `VIDEO` | none | Directory for one MP4 per graded episode |
| `IMAGE` | `256` | Camera frame size, px |
| `POLICY_PORT` | `8000` | SmolVLA server |
| `DETECT`, `DETECT_PORT` | off, `8002` | `1` gives `code` and `supervise` a `detect` tool through the detector server |
| `MEMORY` | none | `code`: a Python file loaded into every program |
| `CAP_POLICY` | off | `code`: `1` adds a `run_policy` primitive (SmolVLA) |
| `LIBERO_POLICY` | `pi05` | `supervise`: `pi05` (Dreamscale) or `smolvla` (local) |
| `LIBERO_TAKEOVER` | `1` | `supervise`: serve the motion tool for taking over |
| `LIBERO_SUITE` | `libero_goal` in the task files | LIBERO suite |
| `LIBERO_PERTURBATION` | none | LIBERO-PRO: `swap` (objects swapped), `task` (new goal) or `lan` (reworded) |
| `LIBERO_TASK_SECONDS` | openpi's budget | Sim seconds per task |
| `TABLETOP_CONTROL` | `ee` | `ee`, `joint` or `delta`; `supervise` and `CAP_POLICY` need `delta` |
| `TABLETOP_GEOMETRY` | `live` | Camera poses published `live`, `static` or `off` |

## Setup

LIBERO needs robosuite 1.4.1, which works only with MuJoCo 3.2.x, so its sim runs in its own venv
that `libero_env/env.py` spawns:

```bash
cd third_party
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git
uv venv --python 3.11 libero-venv
VIRTUAL_ENV=$PWD/libero-venv uv pip install robosuite==1.4.1 mujoco==3.2.7 bddl==1.0.1 \
    future gym==0.25.2 easydict termcolor pyyaml tqdm imageio matplotlib torch scipy \
    "numpy<2" "hud[robot] @ git+https://github.com/kalashjain23/hud-python@feat/robot-control-interfaces"
echo "$PWD/LIBERO" > libero-venv/lib/python3.11/site-packages/libero_src.pth
```

SmolVLA (`lerobot/smolvla_libero`) and the Grounding DINO detector need numpy 2, so they share
another venv and serve over localhost:

```bash
cd third_party && uv venv --python 3.12 smolvla-venv
VIRTUAL_ENV=$PWD/smolvla-venv uv pip install "lerobot[smolvla]" msgpack websockets
VIRTUAL_ENV=$PWD/smolvla-venv uv pip install --no-deps openpi-client
cd ..
third_party/smolvla-venv/bin/python libero_env/smolvla_server.py --port 8000
third_party/smolvla-venv/bin/python perception/detector_server.py --port 8002
```

LIBERO-PRO's files, for `LIBERO_PERTURBATION`:

```bash
hf download zhouxueyang/LIBERO-Pro --repo-type dataset --local-dir third_party/libero-pro \
    --include "bddl_files/libero_goal_*/*" --include "init_files/libero_goal_*/*"
```

`LIBERO_POLICY=pi05` needs a Dreamscale account (`dreamscale login`); each episode opens one
session, billed by the hour.
