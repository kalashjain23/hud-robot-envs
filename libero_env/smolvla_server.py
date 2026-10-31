"""Serve a LeRobot SmolVLA checkpoint over openpi's websocket protocol, for ``RemoteModel``.

LeRobot needs numpy 2 and ``hud[robot]`` pins numpy below 2 (through openpi-client), so the
policy runs in its own venv (``third_party/smolvla-venv``, see the README) and the env reaches it
the way it would any openpi server::

    third_party/smolvla-venv/bin/python libero_env/smolvla_server.py --port 8000

A request is the checkpoint's own input: its image features as HWC uint8 frames (already
oriented as it was trained), ``observation.state``, and ``task``. The reply's ``actions`` is
one unnormalized chunk, ``[T, A]``.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from typing import Any

import numpy as np
import torch
import websockets
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from openpi_client import msgpack_numpy


def load(checkpoint: str, device: str) -> tuple[Any, Any, Any]:
    policy = SmolVLAPolicy.from_pretrained(checkpoint).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocess, postprocess


def infer(policy: Any, preprocess: Any, postprocess: Any, request: dict[str, Any]) -> np.ndarray:
    batch: dict[str, Any] = {"task": request["task"]}
    for key, value in request.items():
        if key.startswith("observation.images."):
            batch[key] = (
                torch.from_numpy(np.ascontiguousarray(value)).permute(2, 0, 1).float() / 255
            )
        elif key == "observation.state":
            batch[key] = torch.from_numpy(np.asarray(value, dtype=np.float32))
    with torch.no_grad():
        chunk = postprocess(policy.predict_action_chunk(preprocess(batch)))
    return chunk[0].float().cpu().numpy()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="lerobot/smolvla_libero")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    policy, preprocess, postprocess = load(args.checkpoint, args.device)
    packer = msgpack_numpy.Packer()

    async def serve(ws: Any) -> None:
        await ws.send(packer.pack({"checkpoint": args.checkpoint}))
        async for message in ws:
            start = time.time()
            actions = await asyncio.to_thread(
                infer, policy, preprocess, postprocess, msgpack_numpy.unpackb(message)
            )
            await ws.send(packer.pack({"actions": actions}))
            print(f"chunk {actions.shape} in {time.time() - start:.2f}s", flush=True)

    async with websockets.serve(serve, "localhost", args.port, compression=None, max_size=None):
        print(f"serving {args.checkpoint} on {args.device}, port {args.port}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
