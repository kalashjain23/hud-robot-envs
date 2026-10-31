"""The policies Policy Supervision runs on LIBERO, as HUD robot ``Model``s and ``Adapter``s.

- pi0.5 fine-tuned on LIBERO, served by Dreamscale: :class:`DreamscaleModel` and
  :class:`Pi05LiberoAdapter`.
- SmolVLA fine-tuned on LIBERO (``lerobot/smolvla_libero``), served locally by
  ``libero_env/smolvla_server.py``: ``RemoteModel`` and :class:`SmolVLALiberoAdapter`.

Both checkpoints were trained on openpi's LIBERO frames, robosuite's render rotated 180
degrees, and on the contract's own 8-dim state and 7-dim delta action.

The pi0.5 adapter turns the contract's observation into Dreamscale's LIBERO observation. The bridge
publishes upright frames, and Dreamscale's client expects robosuite's raw render (it applies
openpi's 180-degree rotation itself), so the frames are flipped back. The chunk is LIBERO's
7-dim delta action, the contract's own, and passes through.

Dreamscale bills a session by the hour while it is open. :class:`DreamscaleModel` opens one on
its first inference and keeps it until :meth:`~DreamscaleModel.close`.
"""

from __future__ import annotations

from typing import Any

import dreamscale
import numpy as np
from dreamscale import libero
from hud.agents.robot import Adapter, Model

MODEL = "pi0.5-libero"
#: The frame size both LIBERO checkpoints were trained on.
POLICY_IMAGE = 256


def policy_frame(frame: Any) -> np.ndarray:
    """A camera frame at the size the policy was trained on: a larger render (rendered for the
    supervising model) is averaged down by whole pixels."""
    frame = np.asarray(frame)
    factor, rest = divmod(frame.shape[0], POLICY_IMAGE)
    if rest or frame.shape[1] != frame.shape[0]:
        raise ValueError(f"frames must be square multiples of {POLICY_IMAGE}, got {frame.shape}")
    if factor == 1:
        return frame
    blocks = frame.reshape(POLICY_IMAGE, factor, POLICY_IMAGE, factor, -1).mean(axis=(1, 3))
    return blocks.round().astype(np.uint8)


class Pi05LiberoAdapter(Adapter):
    """The contract's cameras and 8-dim state as Dreamscale's LIBERO observation."""

    def adapt_observation(self, obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        data = obs["data"]
        return {
            "observation": libero.observation(
                agent_frame=policy_frame(data["observation/agentview"])[::-1],
                wrist_frame=policy_frame(data["observation/wrist"])[::-1],
                state=np.asarray(data["observation/state"]).tolist(),
            ),
            "instruction": prompt,
        }


class SmolVLALiberoAdapter(Adapter):
    """The contract's cameras and state as ``lerobot/smolvla_libero``'s input features."""

    def adapt_observation(self, obs: dict[str, Any], prompt: str) -> dict[str, Any]:
        data = obs["data"]
        # Upright frames to robosuite's raw render ([::-1]), then rotated 180 degrees: a mirror.
        return {
            "observation.images.camera1": np.ascontiguousarray(
                policy_frame(data["observation/agentview"])[:, ::-1]
            ),
            "observation.images.camera2": np.ascontiguousarray(
                policy_frame(data["observation/wrist"])[:, ::-1]
            ),
            "observation.state": np.asarray(data["observation/state"], dtype=np.float32),
            "task": prompt,
        }


class DreamscaleModel(Model):
    """One Dreamscale session per process, opened on first use. Not batchable."""

    def __init__(self, model: str = MODEL) -> None:
        self.model = model
        self._policy: Any = None

    def infer(self, batch: dict[str, Any]) -> np.ndarray:
        if self._policy is None:
            # keep_warm=0: release the GPU the moment the session closes, nothing billed after.
            self._policy = dreamscale.connect(self.model, keep_warm=0, on_progress=None)
        chunk = self._policy.predict(batch["observation"], batch["instruction"])
        return np.asarray(chunk.actions, dtype=np.float32)[None]

    def close(self) -> None:
        if self._policy is None:
            return
        self._policy.close()
        self._policy = None
