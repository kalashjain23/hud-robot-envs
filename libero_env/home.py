"""LIBERO's home pose and closed-loop controllers that drive the arm back to it, or the
fingertips to any point (``reach``).

Every LIBERO-Goal episode starts with the fingertips at about (-0.21, 0.00, 1.17) m, pointing
straight down, gripper open (measured over 8 tasks x 3 init states: within 2 cm and 0.01 rad;
the init states randomize it that much). The controller servos to that pose through the
contract's own delta action, one tick per call, so homing is real arm motion in sim time.
It opens the gripper first (letting go of whatever it holds), rises to the home height, then
moves across and turns, so it does not sweep through the objects on the table. Each tick it
steps the target a capped distance along the straight line to home (an interpolation re-read
from the arm's state every tick, so it slows into home instead of overshooting). A straight
line can still be blocked, by a fixture or by the arm folding into itself near its base.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

HOME_POS = np.array([-0.21, 0.0, 1.172])
HOME_ROTVEC = np.array([3.141, 0.0, -0.09])
#: The OSC_POSE action's scale: one unit commands this displacement per step.
METRES, RADIANS = 0.05, 0.5
#: The most the target moves toward home per tick (20 Hz): 0.4 m/s, 2 rad/s.
STEP, TURN = 0.02, 0.1
POS_TOL, ROT_TOL = 0.01, 0.05
RISE = 0.03  # this far below home, rise in place before moving across
FINGERS_OPEN = 0.035


def matrix(rotvec: NDArray[np.float64]) -> NDArray[np.float64]:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-9:
        return np.eye(3)
    x, y, z = rotvec / angle
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def rotvec(m: NDArray[np.float64]) -> NDArray[np.float64]:
    angle = np.arccos(np.clip((np.trace(m) - 1) / 2, -1.0, 1.0))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    return axis / (2 * np.sin(angle)) * angle


def home(obs: dict[str, Any]) -> tuple[NDArray[np.float64], None, set[str]]:
    """A motion planner: the next few delta actions toward home, none once it is there."""
    state = np.asarray(obs["data"]["observation/state"], dtype=np.float64)
    position = HOME_POS - state[:3]
    rotation = rotvec(matrix(HOME_ROTVEC) @ matrix(state[3:6]).T)  # world frame
    if min(state[6], -state[7]) < FINGERS_OPEN:
        return np.r_[np.zeros(6), -1.0][None], None, set()
    if position[2] > RISE:
        position[:2] = 0.0
        rotation[:] = 0.0
    elif np.linalg.norm(position) < POS_TOL and np.linalg.norm(rotation) < ROT_TOL:
        return np.zeros((0, 7)), None, set()
    step = position * min(1.0, STEP / max(float(np.linalg.norm(position)), 1e-9))
    turn = rotation * min(1.0, TURN / max(float(np.linalg.norm(rotation)), 1e-9))
    return np.r_[step / METRES, turn / RADIANS, -1.0][None], None, set()


#: A reach that has not got this much closer in ``STALL_TICKS`` ticks is stuck and stops.
PROGRESS, STALL_TICKS = 0.002, 10


def reach(target: NDArray[np.float64]) -> Any:
    """A motion planner for the fingertips to the world point *target* (m): the next delta
    action toward it along a straight line, orientation and grip held, none once it is there or
    has stopped getting closer (blocked by a fixture, an object or the arm's own reach)."""
    best = np.inf
    since = 0

    def plan(obs: dict[str, Any]) -> tuple[NDArray[np.float64], None, set[str]]:
        nonlocal best, since
        state = np.asarray(obs["data"]["observation/state"], dtype=np.float64)
        position = np.asarray(target, dtype=np.float64) - state[:3]
        distance = float(np.linalg.norm(position))
        if distance < best - PROGRESS:
            best, since = distance, 0
        else:
            since += 1
        if distance < POS_TOL or since >= STALL_TICKS:
            return np.zeros((0, 7)), None, set()
        step = position * min(1.0, STEP / distance)
        return np.r_[step / METRES, np.zeros(3), 0.0][None], None, set()

    return plan
