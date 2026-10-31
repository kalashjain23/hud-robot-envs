"""SmolVLA fine-tuned on LIBERO (``lerobot/smolvla_libero``), run zero-shot on the tabletop.

The tabletop is not LIBERO: another scene, another camera layout, another world frame and a
10 Hz delta controller. :class:`LiberoPolicyAdapter` translates what it can, so the policy is
out of distribution in the scene only, not in the wiring. It needs ``TABLETOP_CONTROL=delta``.

- Cameras: scene as LIBERO's agentview, wrist as its eye-in-hand, both mirrored left-right
  (LIBERO checkpoints see robosuite's render rotated 180 degrees, which is upright and mirrored).
- State: the fingertip position moved into LIBERO's frame by the offset between the two homes,
  orientation as axis-angle (both report pointing down as xyzw (1, 0, 0, 0)), and the gripper as
  LIBERO's two finger openings.
- Action: LIBERO's per-step delta at 20 Hz moves the arm about 13 mm per unit (measured on
  LIBERO's OSC_POSE, steady state), so each pair of steps becomes one tabletop tick of 26 mm per
  unit, clipped to the tick's bound. Rotation deltas are dropped (the tabletop's delta mode
  holds the hand pointing down). The gripper's direction (+1 closes) becomes a displacement.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from hud.agents.robot import Adapter

from tabletop.sim import DELTA, HOME

#: The tabletop home (fingertip) and LIBERO's start pose in its own frame (libero_10 task 0).
LIBERO_HOME = np.array([-0.046, -0.004, 0.702])
#: Steady-state travel of LIBERO's arm per unit of action per 20 Hz step, in metres.
LIBERO_M_PER_UNIT = 0.013
#: LIBERO finger opening, open (each finger; the right one is negative).
FINGER_OPEN = 0.039
STEPS_PER_TICK = 2  # LIBERO runs at 20 Hz, the tabletop at 10 Hz


def axis_angle(xyzw: np.ndarray) -> np.ndarray:
    w = float(np.clip(xyzw[3], -1.0, 1.0))
    sine = np.sqrt(1.0 - w * w)
    return np.zeros(3) if np.isclose(sine, 0.0) else xyzw[:3] * 2.0 * np.arccos(w) / sine


class LiberoPolicyAdapter(Adapter):
    """The tabletop's ``delta`` contract as ``lerobot/smolvla_libero``'s input and output."""

    def adapt_observation(self, obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        data = obs["data"]
        state = np.asarray(data["observation/state"], dtype=np.float64)
        position = state[:3] - np.array(HOME) + LIBERO_HOME
        finger = FINGER_OPEN * (1 - state[7]) / 2
        return {
            "observation.images.camera1": np.ascontiguousarray(data["observation/scene"][:, ::-1]),
            "observation.images.camera2": np.ascontiguousarray(data["observation/wrist"][:, ::-1]),
            "observation.state": np.array(
                [*position, *axis_angle(state[3:7]), finger, -finger], dtype=np.float32
            ),
            "task": prompt,
        }

    def adapt_chunk(self, chunk: np.ndarray, obs: dict[str, Any]) -> np.ndarray:
        steps = chunk[: len(chunk) // STEPS_PER_TICK * STEPS_PER_TICK]
        pairs = steps.reshape(-1, STEPS_PER_TICK, steps.shape[-1]).sum(axis=1)
        move = np.clip(pairs[:, :3] * LIBERO_M_PER_UNIT, -DELTA, DELTA)
        grip = np.clip(pairs[:, 6], -1.0, 1.0)
        return np.column_stack([move, grip]).astype(np.float32)
