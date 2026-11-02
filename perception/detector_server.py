"""Serve Grounding DINO, an open-vocabulary detector, over HTTP for a ``detect`` primitive.

It needs torch and transformers, which the env's venv does not carry, so it runs in the
SmolVLA venv beside the policy server::

    third_party/smolvla-venv/bin/python perception/detector_server.py --port 8002

``POST /detect`` takes ``{"image": <base64 PNG>, "prompt": "red cube", "threshold": 0.3}`` and
answers ``{"detections": [{"label", "score", "box": [u0, v0, u1, v1]}]}``, boxes in pixels.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


class GroundingDino:
    def __init__(self, checkpoint: str, device: str) -> None:
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint).to(device)
        self.device = device
        self.lock = Lock()  # one image through the model at a time

    def detect(self, image: Image.Image, prompt: str, threshold: float) -> list[dict[str, Any]]:
        # Grounding DINO wants lower-case phrases, each ending with a period.
        text = prompt.lower().strip().rstrip(".") + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with self.lock, torch.no_grad():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        # With no box above the threshold, transformers still returns one (empty) label.
        if not len(result["boxes"]):
            return []
        return [
            {
                "label": label,
                "score": round(float(score), 3),
                "box": [round(float(v), 1) for v in box],
            }
            for label, score, box in zip(
                result["text_labels"], result["scores"], result["boxes"], strict=True
            )
        ]


CHECKPOINT = "IDEA-Research/grounding-dino-tiny"


def serve(detector: GroundingDino, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            image = Image.open(io.BytesIO(base64.b64decode(request["image"]))).convert("RGB")
            detections = detector.detect(
                image, request["prompt"], float(request.get("threshold", 0.3))
            )
            body = json.dumps({"detections": detections}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    serve(GroundingDino(CHECKPOINT, args.device), args.port)


if __name__ == "__main__":
    main()
