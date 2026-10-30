"""A MuJoCo tabletop with a Franka Panda (MuJoCo Menagerie), two cubes and two plates.

Each seed lays the objects out and picks which cube goes on which plate:

- ``lift``       - pick up one named cube.
- ``place``      - put one named cube on one named plate; the other pair is a distractor.
- ``place_both`` - put each named cube on its named plate, one pair after the other. Scored
  by the share of pairs placed.

One scene, three control levels, picked per sim by ``control``:

- ``ee``    - absolute end-effector pose ``[x, y, z, qx, qy, qz, qw, gripper]``
  (``EE_ABS_POSE``). The bridge solves IK every tick and commands the joints.
- ``joint`` - absolute joint angles ``[joint1..joint7, gripper]`` (``JOINT_ABS_POS``),
  straight to the arm's joint position servos.
- ``delta`` - end-effector displacement per tick ``[dx, dy, dz, gripper]``
  (``EE_DEL_POS``); orientation holds pointing down. The gripper is a displacement too.

The world frame has the robot base at the origin and the table top at z = 0. ``gripper``
is -1 open to +1 closed in every mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import av
import mujoco
import numpy as np
from hud.environment.robot import RobotBridge
from numpy.typing import NDArray

Control = Literal["ee", "joint", "delta"]
Geometry = Literal["off", "static", "live"]
Wrist = Literal["down", "tilted"]

SCENE = Path(__file__).resolve().parents[1] / "assets" / "franka_emika_panda" / "tabletop.xml"
CONTROL_RATE = 10
#: Default camera image size (px, square).
IMAGE = 256
CAMERAS = ("scene", "wrist")
#: Reach box for end-effector targets (m).
LOW = [0.25, -0.35, 0.0]
HIGH = [0.75, 0.35, 0.45]
#: Largest end-effector step per tick in ``delta`` mode (m).
DELTA = 0.02
#: Fingertip centre in the hand frame, and the hand's pointing-down orientation (wxyz).
TCP = [0.0, 0.0, 0.1034]
DOWN = np.array([0.0, 1.0, 0.0, 0.0])
HOME = [0.5, 0.0, 0.3]
ARM = [f"joint{i}" for i in range(1, 8)]
FINGER_OPEN = 0.04  # each finger's slide (m) when fully open
GRIPPER_OPEN = 255.0  # actuator8 command when fully open
POSE = ["x", "y", "z", "qx", "qy", "qz", "qw"]
EE_NAMES = ["eef.x", "eef.y", "eef.z", "eef_quat.qx", "eef_quat.qy", "eef_quat.qz", "eef_quat.qw"]

Task = Literal["lift", "place", "place_both"]
CUBES = ("red", "blue")
PLATES = ("green", "yellow")
#: A cube counts as on a plate within this distance of its centre (the plate's radius is 5.5 cm).
ON_PLATE = 0.045


@dataclass(frozen=True)
class Layout:
    """Where an episode's objects start (x, y), and the (cube, plate) pairs it asks for."""

    cubes: dict[str, NDArray[np.float64]]
    plates: dict[str, NDArray[np.float64]]
    pairs: list[tuple[str, str]]


def layout(task: Task, seed: int) -> Layout:
    """*seed*'s layout: plates on either side at the back, cubes apart in front, and a random
    pairing. ``lift`` and ``place`` ask for the first pair only."""
    rng = np.random.default_rng(seed)
    sides = rng.permutation([0.15, -0.15])
    plates = {
        plate: np.array([rng.uniform(0.58, 0.66), side + rng.uniform(-0.03, 0.03)])
        for plate, side in zip(PLATES, sides, strict=True)
    }
    while True:
        spots = rng.uniform([0.38, -0.15], [0.52, 0.15], size=(2, 2))
        if np.linalg.norm(spots[0] - spots[1]) > 0.09:
            break
    cubes = dict(zip(CUBES, spots, strict=True))
    pairs = list(
        zip(rng.permutation(CUBES).tolist(), rng.permutation(PLATES).tolist(), strict=True)
    )
    return Layout(cubes, plates, pairs if task == "place_both" else pairs[:1])


def prompt(task: Task, pairs: list[tuple[str, str]]) -> str:
    (cube, plate), *rest = pairs
    if task == "lift":
        return f"Pick up the {cube} cube and lift it at least 10 cm above the table."
    text = f"Put the {cube} cube on the {plate} plate"
    return f"{text}, then the {rest[0][0]} cube on the {rest[0][1]} plate." if rest else f"{text}."


#: Wrist camera mounts on the hand (pos, quat wxyz); image up is hand +x in both.
WRIST_MOUNTS: dict[Wrist, tuple[list[float], list[float]]] = {
    # Beside the hand, looking straight along the fingers: a top-down view when the hand
    # points down.
    "down": ([0.07, 0.0, 0.0], [0.0, 0.7071068, 0.7071068, 0.0]),
    # Off the side of the hand and tilted 40 degrees in, as wrist cameras are on real arms
    # (DROID, LeRobot): the fingers and anything they hold sit at the bottom of the image, the
    # workspace ahead above them. Looks at a point 5 cm past the fingertip.
    "tilted": ([0.10, 0.0, 0.03], [0.2407, 0.6649, 0.6649, 0.2407]),
}


def build_model(wrist: Wrist = "tilted", image: int = IMAGE) -> mujoco.MjModel:
    """The scene, with a fingertip site and a wrist camera added to the hand, rendering
    *image*-pixel frames."""
    spec = mujoco.MjSpec.from_file(str(SCENE))
    spec.visual.global_.offwidth = spec.visual.global_.offheight = max(image, 640)
    hand = spec.body("hand")
    hand.add_site(name="tcp", pos=TCP, size=[0.004, 0, 0], rgba=[1, 1, 0, 0])
    pos, quat = WRIST_MOUNTS[wrist]
    hand.add_camera(name="wrist", pos=pos, fovy=75, quat=quat)
    # Menagerie's finger servo squeezes a 4 cm cube with ~2 N; 5x stiffer holds it like the
    # real hand (same 0..255 command, same open width).
    gripper = spec.actuator("actuator8")
    gripper.gainprm[0] *= 5
    gripper.biasprm[1:3] = [-500, -20]
    return spec.compile()


def mount(model: mujoco.MjModel, name: str, geometry: Geometry) -> dict[str, Any]:
    """A camera's contract ``transform`` and ``intrinsics``, read off the model.

    A camera fixed in the world gets a static pose. One on a moving body gets a live pose,
    reported each tick as ``observation/<name>_pose``, or with ``static`` geometry its fixed
    pose on that body. ``off`` publishes nothing.
    """
    if geometry == "off":
        return {}
    camera = model.camera(name)
    intrinsics = {"fovy": float(camera.fovy[0])}
    parent = model.body(camera.bodyid[0]).name
    if parent != "world" and geometry == "live":
        return {"transform": {"pose": f"observation/{name}_pose"}, "intrinsics": intrinsics}
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, camera.quat)
    return {
        "transform": {
            "parent": parent,
            "translation": camera.pos.round(4).tolist(),
            "rotation": optical_xyzw(rotation.reshape(3, 3)),
        },
        "intrinsics": intrinsics,
    }


def optical_xyzw(rotation: NDArray[np.float64]) -> list[float]:
    """A MuJoCo camera orientation as the contract's optical frame, xyzw. MuJoCo cameras look
    along -z with +y up, so both axes flip."""
    wxyz = np.zeros(4)
    mujoco.mju_mat2Quat(wxyz, (rotation @ np.diag([1.0, -1.0, -1.0])).flatten())
    return np.roll(wxyz, -1).round(4).tolist()


def contract(
    control: Control,
    joint_low: list[float],
    joint_high: list[float],
    mounts: dict[str, dict[str, Any]],
    size: int = IMAGE,
) -> dict[str, Any]:
    """Cameras (with their *mounts*) plus two state vectors: the control space (the one as wide
    as the action, which direct control reads as the current pose) and the other view of the
    arm."""
    image = {"role": "observation", "type": "rgb", "shape": [size, size, 3], "dtype": "uint8"}
    ee_state = [*EE_NAMES, "gripper"]
    joint_state = [*ARM, "gripper"]
    if control == "joint":
        state, other, other_names = joint_state, "observation/eef", EE_NAMES
        action = {
            "state_type": "JOINT_ABS_POS",
            "names": joint_state,
            "limits": {"min": [*joint_low, -1.0], "max": [*joint_high, 1.0]},
        }
    elif control == "ee":
        state, other, other_names = ee_state, "observation/joints", ARM
        action = {
            "state_type": "EE_ABS_POSE",
            "names": ee_state,
            "limits": {"min": [*LOW, -1, -1, -1, -1, -1.0], "max": [*HIGH, 1, 1, 1, 1, 1.0]},
        }
    else:
        state, other, other_names = ee_state, "observation/joints", ARM
        action = {
            "state_type": "EE_DEL_POS",
            "names": ["eef.dx", "eef.dy", "eef.dz", "gripper"],
            "limits": {"min": [-DELTA] * 3 + [-1.0], "max": [DELTA] * 3 + [1.0]},
        }
    return {
        "robot_type": "franka_panda",
        "control_rate": CONTROL_RATE,
        "features": {
            **{f"observation/{camera}": {**image, **mounts[camera]} for camera in CAMERAS},
            **{
                live: {"role": "observation", "shape": [7], "dtype": "float32", "names": POSE}
                for mount in mounts.values()
                if (live := mount.get("transform", {}).get("pose"))
            },
            "observation/state": {
                "role": "observation",
                "shape": [len(state)],
                "dtype": "float32",
                "names": state,
            },
            other: {
                "role": "observation",
                "shape": [len(other_names)],
                "dtype": "float32",
                "names": other_names,
            },
            "action": {
                "role": "action",
                "shape": [len(action["names"])],
                "dtype": "float32",
                **action,
            },
        },
    }


class Franka(RobotBridge):
    """One tabletop episode per reset; ``task`` picks the goal, ``seed`` the cube's spot.

    With ``video`` set to a directory, each episode's camera frames (one per control tick, all
    cameras side by side) are written there as an MP4 when it is graded. ``geometry`` picks how
    camera mounts are published (``static`` and ``off`` only to measure what they are worth).
    """

    def __init__(
        self,
        control: Control = "ee",
        video: str = "",
        geometry: Geometry = "live",
        wrist: Wrist = "tilted",
        image: int = IMAGE,
    ) -> None:
        super().__init__()
        self.control = control
        self.video = video
        self.frames: list[NDArray[np.uint8]] = []
        self.ticks = 0
        self.seed = 0
        self.model = build_model(wrist, image)
        self.data = mujoco.MjData(self.model)
        self.scratch = mujoco.MjData(self.model)  # IK solves here, never stepped
        self.renderer = mujoco.Renderer(self.model, image, image)
        self.substeps = round(1 / (CONTROL_RATE * self.model.opt.timestep))
        joints = [self.model.joint(name) for name in ARM]
        self.arm = np.array([joint.qposadr[0] for joint in joints])
        self.dofs = np.array([joint.dofadr[0] for joint in joints])
        self.joint_low, self.joint_high = np.array([joint.range for joint in joints]).T
        self.tcp = self.model.site("tcp").id
        self.cubes = {cube: self.model.body(f"cube_{cube}").id for cube in CUBES}
        self.plates = {plate: self.model.body(f"plate_{plate}").id for plate in PLATES}
        self.layout = layout("lift", 0)
        self.finger = self.model.joint("finger_joint1").qposadr[0]
        self.home_q = self.model.key("home").qpos[self.arm].copy()
        self.contract = contract(
            control,
            self.joint_low.tolist(),
            self.joint_high.tolist(),
            {camera: mount(self.model, camera, geometry) for camera in CAMERAS},
            image,
        )
        # Cameras on a moving body, whose pose each observation reports.
        self.live = [
            camera
            for camera in CAMERAS
            if "pose" in self.contract["features"][f"observation/{camera}"].get("transform", {})
        ]
        self.task: Task = "lift"
        # The commanded fingertip and gripper, which delta mode displaces. Displacing the
        # measured pose instead would compound the servo's tracking error every tick.
        self.target = np.array(HOME)
        self.grip = -1.0

    def reset(self, task: Task = "lift", seed: int = 0) -> str:
        self.task = task
        self.seed = seed
        self.layout = layout(task, seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        for plate, xy in self.layout.plates.items():
            self.model.body_pos[self.plates[plate], :2] = xy
        for cube, xy in self.layout.cubes.items():
            self.data.joint(f"cube_{cube}").qpos[:7] = [*xy, 0.02, 1, 0, 0, 0]
        q = self.ik(np.array(HOME), DOWN, self.home_q)
        self.data.qpos[self.arm] = q
        self.data.ctrl[:7] = q
        self.data.ctrl[7] = GRIPPER_OPEN
        self.target = np.array(HOME)
        self.grip = -1.0
        mujoco.mj_forward(self.model, self.data)
        self.success = False
        self.terminated = False
        self.frames.clear()
        self.ticks = 0
        return prompt(task, self.layout.pairs)

    def step(self, action: NDArray[Any]) -> None:
        command = np.asarray(action[0], dtype=np.float64)
        if self.control == "joint":
            arm, self.grip = command[:7], command[7]
        elif self.control == "ee":
            xyzw = command[3:7] / np.linalg.norm(command[3:7])
            arm = self.ik(command[:3], np.roll(xyzw, 1), self.data.ctrl[:7])
            self.grip = command[7]
        else:
            self.target = np.clip(self.target + command[:3], LOW, HIGH)
            arm = self.ik(self.target, DOWN, self.data.ctrl[:7])
            self.grip = float(np.clip(self.grip + command[3], -1, 1))
        self.data.ctrl[:7] = np.clip(arm, self.joint_low, self.joint_high)
        self.data.ctrl[7] = (1 - np.clip(self.grip, -1, 1)) / 2 * GRIPPER_OPEN
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        self.ticks += 1
        self.success = self._succeeded()
        self.terminated = self.success

    def ik(
        self, pos: NDArray[np.float64], wxyz: NDArray[np.float64], start: NDArray[Any]
    ) -> NDArray[np.float64]:
        """Joint angles putting the fingertip at *pos* with orientation *wxyz*: damped least
        squares from *start*, pulled toward the home posture in the null space."""
        data = self.scratch
        data.qpos[:] = self.data.qpos
        q = np.array(start, dtype=np.float64)
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        current, inverse, delta = np.zeros(4), np.zeros(4), np.zeros(4)
        error = np.zeros(6)
        for _ in range(30):
            data.qpos[self.arm] = q
            mujoco.mj_kinematics(self.model, data)
            mujoco.mj_comPos(self.model, data)
            error[:3] = pos - data.site_xpos[self.tcp]
            mujoco.mju_mat2Quat(current, data.site_xmat[self.tcp])
            mujoco.mju_negQuat(inverse, current)
            mujoco.mju_mulQuat(delta, wxyz, inverse)
            mujoco.mju_quat2Vel(error[3:], delta, 1.0)
            if np.linalg.norm(error[:3]) < 1e-4 and np.linalg.norm(error[3:]) < 1e-3:
                break
            mujoco.mj_jacSite(self.model, data, jacp, jacr, self.tcp)
            jac = np.vstack([jacp, jacr])[:, self.dofs]
            step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), error)
            null = np.eye(7) - np.linalg.pinv(jac) @ jac
            q = np.clip(q + step + 0.1 * null @ (self.home_q - q), self.joint_low, self.joint_high)
        return q

    def _succeeded(self) -> bool:
        if self.task == "lift":
            return bool(self.data.xpos[self.cubes[self.layout.pairs[0][0]], 2] > 0.12)
        return self._placed() == len(self.layout.pairs) and self._grip() < 0

    def _placed(self) -> int:
        """Pairs whose cube rests on its plate (centre within reach of the plate's middle, low
        enough to sit on it rather than be held above it)."""
        placed = 0
        for cube, plate in self.layout.pairs:
            position = self.data.xpos[self.cubes[cube]]
            middle = self.data.xpos[self.plates[plate], :2]
            placed += bool(np.linalg.norm(position[:2] - middle) < ON_PLATE and position[2] < 0.045)
        return placed

    def _grip(self) -> float:
        """Finger opening as -1 (open) to +1 (closed)."""
        return float(1 - 2 * self.data.qpos[self.finger] / FINGER_OPEN)

    def get_observation(self) -> tuple[dict[str, NDArray[Any]], NDArray[Any]]:
        data: dict[str, NDArray[Any]] = {}
        for camera in CAMERAS:
            self.renderer.update_scene(self.data, camera=camera)
            data[f"observation/{camera}"] = self.renderer.render()[None]
        for camera in self.live:
            view = self.model.camera(camera).id
            pose = [
                *self.data.cam_xpos[view],
                *optical_xyzw(self.data.cam_xmat[view].reshape(3, 3)),
            ]
            data[f"observation/{camera}_pose"] = np.array([pose], dtype=np.float32)
        # One frame per tick: a direct-control call also observes before its first step.
        if self.video and len(self.frames) <= self.ticks:
            self.frames.append(np.hstack([data[f"observation/{c}"][0] for c in CAMERAS]))
        wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(wxyz, self.data.site_xmat[self.tcp])
        ee = [*self.data.site_xpos[self.tcp], *np.roll(wxyz, -1)]
        joints = list(self.data.qpos[self.arm])
        grip = self._grip()
        if self.control == "joint":
            data["observation/state"] = np.array([[*joints, grip]], dtype=np.float32)
            data["observation/eef"] = np.array([ee], dtype=np.float32)
        else:
            data["observation/state"] = np.array([[*ee, grip]], dtype=np.float32)
            data["observation/joints"] = np.array([joints], dtype=np.float32)
        return data, np.array([self.terminated])

    def result(self) -> dict[str, Any]:
        if self.video and self.frames:
            self._write_video(Path(self.video) / f"{self.task}-seed{self.seed}.mp4")
        # Placing tasks earn a share per pair placed; lift is all or nothing.
        score = self._placed() / len(self.layout.pairs) if self.task != "lift" else 0.0
        return {**super().result(), "score": 1.0 if self.success else score}

    def _write_video(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264", rate=CONTROL_RATE)
            stream.height, stream.width = self.frames[0].shape[:2]
            stream.pix_fmt = "yuv420p"
            for frame in self.frames:
                container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
            container.mux(stream.encode())
