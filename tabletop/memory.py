"""Code as Policy memory from solved tabletop episodes: pick a cube, place it on a plate. Loaded
into every program with ``TABLETOP_MEMORY=tabletop/memory.py`` (``TABLETOP_CONTROL=delta``); the
primitives are already in scope."""

#: Fingertip heights in metres: above the cubes, closing on a cube, carrying, releasing.
HOVER, GRASP, CARRY, RELEASE = 0.09, 0.018, 0.15, 0.06


def fingertip():
    """The fingertip's (x, y, z) in the world frame and the gripper reading: -1 open, about 0
    closed on a cube, near +1 closed on nothing."""
    s = observe().state["state"]
    return s["eef.x"], s["eef.y"], s["eef.z"], s["gripper"]


def go(x=None, y=None, z=None, tol=0.003, steps=60):
    """Move the fingertip to absolute world coordinates (None keeps that axis), correcting from
    the reported position until within tol metres. Returns where it ended, (x, y, z)."""
    for _ in range(steps):
        px, py, pz, _ = fingertip()
        d = [0.0 if t is None else t - p for t, p in ((x, px), (y, py), (z, pz))]
        if max(abs(v) for v in d) < tol:
            break
        move_by(x=d[0], y=d[1], z=d[2])
    return fingertip()[:3]


def close_gripper():
    """Close the hand and let the fingers settle. Returns the gripper reading."""
    move_by(grip=2)
    wait(1.0)
    return fingertip()[3]


def open_gripper():
    """Open the hand fully and let the fingers settle."""
    move_by(grip=-2)
    wait(1.0)


def grasp(x, y, lower=0.0):
    """Pick up the cube centred at world (x, y): hover, descend, close, lift to carry height.
    Returns True if the cube is held."""
    open_gripper()
    go(z=HOVER)
    go(x, y, HOVER)
    go(z=GRASP - lower)
    close_gripper()
    go(z=CARRY)
    return fingertip()[3] < 0.6


def place(x, y):
    """Carry the held cube over world (x, y), lower it, release it and back off upward."""
    go(z=CARRY)
    go(x, y, CARRY)
    go(z=RELEASE)
    open_gripper()
    go(z=CARRY)


def regrasp(x, y, lower=0.006):
    """Recovery skill. Failure: grasp() returned False, the hand closed on nothing (gripper
    reading near +1), usually because it closed above the cube or beside it. When: right after
    a failed grasp, once you have re-estimated the cube's centre from a fresh look. Fix: reopen,
    re-centre over (x, y) and close a few mm lower. Returns True if the cube is now held."""
    return grasp(x, y, lower)
