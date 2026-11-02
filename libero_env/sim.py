"""LIBERO behind a ``robot`` wire, set up as the benchmark runs it.

Runs in LIBERO's own venv (robosuite 1.4.1, MuJoCo 3.2.7; see the README), as a bridge
child: ``python -m hud.environment.robot.bridge libero_env/sim.py:Libero``.

What is LIBERO's, unchanged:

- Tasks, scenes, language instructions and the 50 fixed init states per task, for one suite
  (``libero_10`` by default). Success is LIBERO's own check.
- The Franka Panda with robosuite's operational-space controller (``OSC_POSE``) at 20 Hz. The
  7-dim action is ``[dx, dy, dz, drx, dry, drz, gripper]`` in [-1, 1]: the arm part is a
  per-step displacement target scaled to 5 cm and 0.5 rad, and the gripper is a direction
  (+1 closes, -1 opens, 0 holds).
- The ``agentview`` and ``robot0_eye_in_hand`` cameras at 256 px, and the episode protocol
  openpi evaluates with: 10 settling steps with the gripper open after each reset.

A chain asks for several of a suite's tasks in one episode, for suites whose tasks share one
scene (``libero_goal``): the episode starts from the first task's init state, the prompt
lists every task in order, and it is graded with LIBERO's own goal predicates for each task,
all checked on the live sim. Its score is the share of the tasks' goals that hold at the end;
it succeeds (and ends) when all hold at once.

What this file adds is the contract: the cameras' mounts (the wrist's as a live pose), and the
proprio openpi feeds pi0 on LIBERO (fingertip position, axis-angle orientation, finger joints).
Frames are flipped upright (robosuite renders bottom-up), so the published camera frames are
the true ones. A VLA trained on openpi's LIBERO frames (the raw render rotated 180 degrees)
applies its own flip.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import zipfile
from pathlib import Path
from typing import Any

THIRD_PARTY = Path(__file__).resolve().parents[1] / "third_party"
# LIBERO reads its paths from a config file, and prompts for one on first import without it.
# Point it at the clone's own files.
_PACKAGE = THIRD_PARTY / "LIBERO" / "libero" / "libero"
_CONFIG = THIRD_PARTY / "libero-config"
_CONFIG.mkdir(exist_ok=True)
(_CONFIG / "config.yaml").write_text(
    "".join(
        f"{key}: {path}\n"
        for key, path in {
            "benchmark_root": _PACKAGE,
            "bddl_files": _PACKAGE / "bddl_files",
            "init_states": _PACKAGE / "init_files",
            "datasets": _PACKAGE.parent / "datasets",
            "assets": _PACKAGE / "assets",
        }.items()
    )
)
os.environ["LIBERO_CONFIG_PATH"] = str(_CONFIG)

import av
import mujoco
import numpy as np
from hud.environment.robot import RobotBridge
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.bddl_utils import robosuite_parse_problem
from numpy.typing import NDArray

CONTROL_RATE = 20
IMAGE = 256
#: Contract camera name -> LIBERO camera.
CAMERAS = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}
#: Steps openpi waits after a reset for objects to settle, gripper open (not scored).
SETTLE_STEPS = 10
SETTLE_ACTION = [0.0] * 6 + [-1.0]
STATE = ["eef.x", "eef.y", "eef.z", "eef.rx", "eef.ry", "eef.rz", "finger.left", "finger.right"]
ACTION = ["eef.dx", "eef.dy", "eef.dz", "eef.drx", "eef.dry", "eef.drz", "gripper"]
POSE = ["x", "y", "z", "qx", "qy", "qz", "qw"]
HORIZON = 10_000
#: LIBERO-PRO's perturbed files (``third_party/libero-pro``, from the ``zhouxueyang/LIBERO-Pro``
#: dataset): ``swap`` moves objects to each other's places, ``task`` changes the goal and
#: ``lan`` rewords the instruction. The last two carry their own instruction.
PERTURBATIONS = ("swap", "task", "lan")
#: What makes a BDDL problem's scene; chained tasks must agree on all of it.
SCENE_KEYS = ("fixtures", "regions", "objects", "scene_properties", "initial_state")


def optical_xyzw(rotation: NDArray[np.float64]) -> list[float]:
    """A MuJoCo camera orientation as the contract's optical frame, xyzw. MuJoCo cameras look
    along -z with +y up, so both axes flip."""
    wxyz = np.zeros(4)
    mujoco.mju_mat2Quat(wxyz, (rotation @ np.diag([1.0, -1.0, -1.0])).flatten())
    return np.roll(wxyz, -1).round(4).tolist()


def axis_angle(xyzw: NDArray[np.float64]) -> NDArray[np.float64]:
    """A unit quaternion (xyzw) as an axis-angle vector, as openpi's LIBERO client computes it."""
    w = float(np.clip(xyzw[3], -1.0, 1.0))
    sine = np.sqrt(1.0 - w * w)
    if np.isclose(sine, 0.0):
        return np.zeros(3)
    return xyzw[:3] * 2.0 * np.arccos(w) / sine


class _NumpyUnpickler(pickle.Unpickler):
    """Plain numpy arrays only: the init files are torch archives around a pickled array, and
    LIBERO-PRO's use pickle protocol 4, which torch's own safe loader rejects."""

    ALLOWED = frozenset(
        {
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy", "ndarray"),
            ("numpy", "dtype"),
            ("_codecs", "encode"),  # protocol 2's way of writing bytes
        }
    )

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"{module}.{name} is not allowed in an init file")
        return super().find_class(module, name)


def _init_states(path: Path) -> NDArray[np.float64]:
    with zipfile.ZipFile(path) as archive:
        (name,) = (n for n in archive.namelist() if n.endswith("/data.pkl"))
        return np.asarray(_NumpyUnpickler(io.BytesIO(archive.read(name))).load())


class Libero(RobotBridge):
    """One LIBERO episode per reset: ``task_id`` picks the suite's task, ``init_state`` one of
    its fixed initial states. With ``video`` set to a directory, each episode's frames (both
    cameras side by side, one per step) are written there as an MP4 when it is graded."""

    def __init__(
        self, suite: str = "libero_10", video: str = "", image: int = IMAGE, perturbation: str = ""
    ) -> None:
        super().__init__()
        if perturbation and perturbation not in PERTURBATIONS:
            raise ValueError(f"perturbation must be one of {PERTURBATIONS}, got {perturbation!r}")
        self.perturbation = perturbation
        self.suite = benchmark.get_benchmark_dict()[suite]()
        self.suite_name = suite
        self.video = video
        self.image = image
        self.env: OffScreenRenderEnv | None = None
        self.task_id = -1
        self.init_state = 0
        self.obs: dict[str, Any] = {}
        self.frames: list[NDArray[np.uint8]] = []
        self.ticks = 0
        #: A chain's tasks and each one's goal predicates (empty: one task, LIBERO's own done).
        self.chain: list[int] = []
        self.goals: list[list[list[str]]] = []
        #: A chain's goal changes, as (tick, task id, holds), for analysis after the episode.
        self.goal_log: list[tuple[int, int, bool]] = []
        self.held: list[bool] = []
        self.contract = self._contract()

    def _open(self, task_id: int) -> OffScreenRenderEnv:
        if self.env is None or task_id != self.task_id:
            if self.env is not None:
                self.env.close()
            task = self.suite.get_task(task_id)
            bddl = self._bddl(task)
            # robosuite ends an episode at its horizon (LIBERO's default 1,000 steps) and then
            # refuses actions; the env's time limit bounds episodes, so set it past any chain.
            self.env = OffScreenRenderEnv(
                bddl_file_name=str(bddl),
                camera_heights=self.image,
                camera_widths=self.image,
                camera_depths=True,
                horizon=HORIZON,
            )
            self.env.seed(0)
            self.task_id = task_id
        return self.env

    @property
    def model(self) -> mujoco.MjModel:
        assert self.env is not None
        return self.env.env.sim.model._model

    @property
    def data(self) -> mujoco.MjData:
        assert self.env is not None
        return self.env.env.sim.data._data

    def _contract(self) -> dict[str, Any]:
        self._open(0)
        image = {
            "role": "observation",
            "type": "rgb",
            "shape": [self.image, self.image, 3],
            "dtype": "uint8",
        }
        cameras: dict[str, Any] = {}
        for name, camera in CAMERAS.items():
            view = self.model.camera(camera)
            intrinsics = {"fovy": float(view.fovy[0])}
            if self.model.body(view.bodyid[0]).name == "world":
                rotation = np.zeros(9)
                mujoco.mju_quat2Mat(rotation, view.quat)
                transform = {
                    "parent": "world",
                    "translation": view.pos.round(4).tolist(),
                    "rotation": optical_xyzw(rotation.reshape(3, 3)),
                }
            else:
                transform = {"pose": f"observation/{name}_pose"}
            cameras[f"observation/{name}"] = {
                **image,
                "transform": transform,
                "intrinsics": intrinsics,
                "depth": f"observation/{name}_depth",
            }
        live = {
            pose: {"role": "observation", "shape": [7], "dtype": "float32", "names": POSE}
            for feature in cameras.values()
            if (pose := feature["transform"].get("pose"))
        }
        # Each camera's depth image: metres along its optical axis, aligned with its frame.
        depths = {
            feature["depth"]: {
                "role": "observation",
                "type": "depth",
                "shape": [self.image, self.image],
                "dtype": "float32",
            }
            for feature in cameras.values()
        }
        return {
            "robot_type": "panda",
            "control_rate": CONTROL_RATE,
            "features": {
                **cameras,
                **depths,
                **live,
                "observation/state": {
                    "role": "observation",
                    "shape": [len(STATE)],
                    "dtype": "float32",
                    "names": STATE,
                },
                "action": {
                    "role": "action",
                    "shape": [len(ACTION)],
                    "dtype": "float32",
                    "names": ACTION,
                    "state_type": "EE_DEL_POSE",
                    "limits": {"min": [-1.0] * len(ACTION), "max": [1.0] * len(ACTION)},
                },
            },
        }

    def reset(self, task_id: int = 0, init_state: int = 0, chain: list[int] | None = None) -> str:
        """One task (``task_id``), or a ``chain`` of the suite's tasks in one scene, started
        from the first one's ``init_state``."""
        self.chain = list(chain or [])
        if self.chain:
            task_id = self.chain[0]
            problems = [
                robosuite_parse_problem(str(self._bddl(self.suite.get_task(i)))) for i in self.chain
            ]
            if any(p[key] != problems[0][key] for p in problems for key in SCENE_KEYS):
                raise ValueError(f"chain {self.chain}: the tasks do not share one scene")
            self.goals = [p["goal_state"] for p in problems]
        else:
            self.goals = []
        env = self._open(task_id)
        task = self.suite.get_task(task_id)
        env.reset()
        self.obs = env.set_init_state(_init_states(self._files("init_states", task))[init_state])
        for _ in range(SETTLE_STEPS):
            self.obs, _, _, _ = env.step(SETTLE_ACTION)
        self.init_state = init_state
        self.success = False
        self.terminated = False
        self.frames.clear()
        self.ticks = 0
        self.held = self.done()
        self.goal_log = [(0, t, True) for t, h in zip(self.chain, self.held) if h]
        if not self.chain:
            return self._instruction(task)
        steps = [self._instruction(self.suite.get_task(i)) for i in self.chain]
        text = ", ".join(steps[:-1]) + f", then {steps[-1]}"
        return text[0].upper() + text[1:] + "."

    def _instruction(self, task: Any) -> str:
        """A task's instruction: LIBERO's, or under ``task`` and ``lan`` LIBERO-PRO's own."""
        if self.perturbation in ("task", "lan"):
            return " ".join(robosuite_parse_problem(str(self._bddl(task)))["language_instruction"])
        return str(task.language)

    def _bddl(self, task: Any) -> Path:
        return self._files("bddl_files", task)

    def _files(self, kind: str, task: Any) -> Path:
        """A task's BDDL (``bddl_files``) or initial states (``init_states``): LIBERO's own, or
        LIBERO-PRO's under the perturbation."""
        name = task.bddl_file if kind == "bddl_files" else task.init_states_file
        if self.perturbation:
            folder = "bddl_files" if kind == "bddl_files" else "init_files"
            return (
                THIRD_PARTY
                / "libero-pro"
                / folder
                / f"{task.problem_folder}_{self.perturbation}"
                / name
            )
        return Path(get_libero_path(kind)) / task.problem_folder / name

    def done(self) -> list[bool]:
        """Whether each chained task's goal holds now."""
        assert self.env is not None
        domain = self.env.env
        return [all(domain._eval_predicate(state) for state in goal) for goal in self.goals]

    def step(self, action: NDArray[Any]) -> None:
        assert self.env is not None
        self.obs, _, done, _ = self.env.step(np.asarray(action[0], dtype=np.float64).tolist())
        self.ticks += 1
        if self.chain:
            held = self.done()
            self.goal_log += [
                (self.ticks, task, now)
                for task, now, before in zip(self.chain, held, self.held)
                if now != before
            ]
            self.held = held
        self.success = all(self.held) if self.chain else bool(done)
        self.terminated = self.success

    def _metres(self, depth: NDArray[np.float32]) -> NDArray[np.float32]:
        """MuJoCo's normalised depth buffer as metres along the optical axis (robosuite's
        ``get_real_depth_map``, which needs h5py to import)."""
        extent = self.model.stat.extent
        near = self.model.vis.map.znear * extent
        far = self.model.vis.map.zfar * extent
        return (near / (1 - depth * (1 - near / far))).astype(np.float32)

    def get_observation(self) -> tuple[dict[str, NDArray[Any]], NDArray[Any]]:
        obs = self.obs
        data: dict[str, NDArray[Any]] = {}
        for name, camera in CAMERAS.items():
            data[f"observation/{name}"] = np.ascontiguousarray(obs[f"{camera}_image"][::-1])[None]
            depth = self._metres(obs[f"{camera}_depth"][::-1, :, 0])
            data[f"observation/{name}_depth"] = np.ascontiguousarray(depth)[None]
            if f"observation/{name}_pose" in self.contract["features"]:
                view = self.model.camera(camera).id
                pose = [
                    *self.data.cam_xpos[view],
                    *optical_xyzw(self.data.cam_xmat[view].reshape(3, 3)),
                ]
                data[f"observation/{name}_pose"] = np.array([pose], dtype=np.float32)
        if self.video and len(self.frames) <= self.ticks:
            self.frames.append(np.hstack([data[f"observation/{name}"][0] for name in CAMERAS]))
        state = [
            *obs["robot0_eef_pos"],
            *axis_angle(obs["robot0_eef_quat"]),
            *obs["robot0_gripper_qpos"],
        ]
        data["observation/state"] = np.array([state], dtype=np.float32)
        return data, np.array([self.terminated])

    def result(self) -> dict[str, Any]:
        if self.video and self.frames:
            tasks = "-".join(map(str, self.chain)) if self.chain else str(self.task_id)
            name = f"{self.suite_name}-task{tasks}-init{self.init_state}.mp4"
            self._write_video(Path(self.video) / name)
            if self.chain:
                log = [{"tick": k, "task": t, "holds": h} for k, t, h in self.goal_log]
                (Path(self.video) / name).with_suffix(".goals.json").write_text(json.dumps(log))
            grader = (Path(self.video) / name).with_suffix(".grader.json")
            grader.write_text(json.dumps(self._grader_state()))
        result = super().result()
        if self.chain:
            done = self.done()
            result |= {
                "score": sum(done) / len(done),
                "done": dict(zip(self.chain, done)),
                "goal_log": self.goal_log,
            }
        return result

    def _grader_state(self) -> list[dict[str, Any]]:
        """Each goal predicate at the end, with the world pose of every object it names: why
        LIBERO's check passed or failed."""
        assert self.env is not None
        domain = self.env.env
        report = []
        for task_id in self.chain or [self.task_id]:
            bddl = self._bddl(self.suite.get_task(task_id))
            for predicate in robosuite_parse_problem(str(bddl))["goal_state"]:
                poses = {}
                for name in predicate[1:]:
                    state = domain.object_states_dict.get(name)
                    if state is not None:
                        geom = state.get_geom_state()
                        poses[name] = {
                            "pos": np.asarray(geom["pos"]).round(4).tolist(),
                            "quat": np.asarray(geom["quat"]).round(4).tolist(),
                        }
                holds = bool(domain._eval_predicate(predicate))
                report.append({"task": task_id, "predicate": predicate, "holds": holds, **poses})
        return report

    def _write_video(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264", rate=CONTROL_RATE)
            stream.height, stream.width = self.frames[0].shape[:2]
            stream.pix_fmt = "yuv420p"
            for frame in self.frames:
                container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
            container.mux(stream.encode())
