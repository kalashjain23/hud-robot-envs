"""Code as Policy memory from solved LIBERO episodes: opening a cabinet drawer, putting a bowl on
a plate. Loaded into every program with ``LIBERO_MEMORY=libero_env/memory.py``; the primitives
and ``np`` are already in scope."""


def state():
    """The arm's state: eef.x/y/z (m, world), eef.rx/ry/rz (axis-angle), finger.left/right
    (m of opening, about 0.04 open, 0 closed)."""
    return observe().state["state"]


def go(x=None, y=None, z=None, grip=None, tol=0.004, steps=40):
    """Move the point between the fingertips to absolute world coordinates (None keeps that
    axis), correcting from the reported position until within tol metres. grip, if given, is
    sent every step (+1 keeps a held object squeezed). Returns the final state."""
    for _ in range(steps):
        s = state()
        here = (s["eef.x"], s["eef.y"], s["eef.z"])
        d = [0.0 if t is None else t - p for t, p in zip((x, y, z), here)]
        if max(abs(v) for v in d) < tol:
            break
        c = np.clip(np.array(d) / 0.05 * 1.5, -1, 1)
        move_by(x=c[0], y=c[1], z=c[2], grip=grip)
    return state()


def close_gripper(steps=12):
    """Close the fingers. Returns True if they stopped on something (opening above 2 mm)."""
    for _ in range(steps):
        move_by(grip=1)
    s = state()
    return s["finger.left"] + s["finger.right"] > 0.004


def open_gripper(steps=10):
    """Open the fingers fully."""
    for _ in range(steps):
        move_by(grip=-1)


def _rv2R(v):
    th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _R2rv(R):
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-6:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2 * np.sin(th)) * th


def orient(target, tol=0.05, steps=14):
    """Turn the gripper, closed-loop, to the rotation matrix *target* (columns: the gripper's
    x, y, z axes in the world). The approach direction is the gripper's z column."""
    for _ in range(steps):
        s = state()
        R = _rv2R(np.array([s["eef.rx"], s["eef.ry"], s["eef.rz"]]))
        d = _R2rv(target @ R.T)
        if np.linalg.norm(d) < tol:
            break
        c = np.clip(d / 0.5 * 0.6, -1, 1)
        move_by(roll=c[0], pitch=c[1], yaw=c[2])
    return state()


#: The gripper pointing horizontally along world -y with its fingers vertical: how it faced the
#: cabinet's handles, which are horizontal bars on the cabinet's +y face.
FACE_MINUS_Y = np.column_stack([[1, 0, 0], [0, 0, 1], [0, -1, 0]])


def pull_handle(x, z, y_front, depth=0.03, pull=0.12):
    """Open a drawer by its handle bar at world (x, z), on a cabinet face at y = y_front: face
    the cabinet, line up in front of the bar, advance until it sits between the fingers, close
    and pull toward +y. Returns True if the fingers held the bar."""
    open_gripper()
    go(y=y_front + 0.08)
    orient(FACE_MINUS_Y)
    go(x=x, z=z)
    go(y=y_front - depth)
    held = close_gripper()
    s = state()
    go(y=s["eef.y"] + pull, grip=1)
    return held


def grasp_rim(x, y, z_rim, lift=0.12):
    """Pick a bowl by its rim wall at world (x, y), the fingers straddling the wall: hover,
    descend to z_rim, close, lift. Returns True if the fingers stopped on the rim."""
    open_gripper()
    go(z=z_rim + 0.1)
    go(x=x, y=y)
    go(z=z_rim)
    held = close_gripper()
    s = state()
    go(z=s["eef.z"] + lift, grip=1)
    return held


def place_held(x, y, z_release):
    """Carry the held object over world (x, y), lower it to z_release, open, and back off up."""
    go(x=x, y=y, grip=1, steps=60)
    go(z=z_release, grip=1)
    open_gripper()
    s = state()
    go(z=s["eef.z"] + 0.1)


def regrasp_lower(x, y, z_rim, lower=0.01, tries=3):
    """Recovery skill. Failure: the fingers closed on nothing (finger opening near 0) or the
    object slipped out when lifted, usually because they closed above the rim or beside it.
    When: right after a failed grasp, once the rim's position is re-checked in a fresh look.
    Fix: reopen and try again, each time closing *lower* metres lower. Returns True once
    something is held."""
    for i in range(1, tries + 1):
        if grasp_rim(x, y, z_rim - i * lower):
            return True
    return False
