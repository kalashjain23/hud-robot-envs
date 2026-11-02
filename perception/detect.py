"""The ``detect`` primitive: open-vocabulary detection through ``detector_server.py``."""

from __future__ import annotations

import base64
import io
import json
import urllib.request
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image


def make_detect(port: int) -> Callable[..., list[dict[str, Any]]]:
    """A ``detect`` primitive reaching the detector server on *port*."""

    def detect(image: Any, prompt: str, threshold: float = 0.3) -> list[dict[str, Any]]:
        """detect(image, prompt, threshold=0.3) -> list of detections, best first: find what
        *prompt* names (e.g. "red cube", "plate") in an (H, W, 3) image such as one of
        observe().images. Each detection is a dict: "label", "score" (0-1), "box"
        (u0, v0, u1, v1) in pixels and "center" (u, v), pixels as (column, row) from the
        top-left. Turn a pixel into metres with the same observation's camera:
        obs.cameras[camera].point(u, v) with depth, else .to_world(u, v, z)."""
        buffer = io.BytesIO()
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format="PNG")
        request = json.dumps(
            {
                "image": base64.b64encode(buffer.getvalue()).decode(),
                "prompt": prompt,
                "threshold": threshold,
            }
        ).encode()
        with urllib.request.urlopen(
            urllib.request.Request(
                f"http://127.0.0.1:{port}/detect",
                data=request,
                headers={"Content-Type": "application/json"},
            ),
            timeout=60,
        ) as reply:
            detections = json.loads(reply.read())["detections"]
        for detection in detections:
            u0, v0, u1, v1 = detection["box"]
            detection["center"] = ((u0 + u1) / 2, (v0 + v1) / 2)
        return sorted(detections, key=lambda d: d["score"], reverse=True)

    return detect
