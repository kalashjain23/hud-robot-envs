"""Benchmark an LLM on these envs: headless Claude Code drives the control interface's tools.

    TABLETOP_INTERFACE=code uv run python scripts/bench.py place_both --seeds 0-4 --out runs/x
    LIBERO_INTERFACE=direct uv run python scripts/bench.py libero --task-id 0 --seeds 0-2 \
        --out runs/y

Each episode is a real rollout; the agent is ``claude -p`` given only the env's ``control``
MCP server (no built-in tools). Rollouts cost money, so every episode is capped: the agent is
stopped once ``--max-calls`` tool results have come back (the CLI's ``--max-turns`` does not
count tool calls), and the CLI enforces ``--max-usd`` spend. Each episode is a fresh session
with no skills or slash commands, so nothing carries over. The default model is the cheapest;
pick a frontier one deliberately. Per episode it writes the model's stream
(``<task>-seed<N>.jsonl``), the sim's video, and a summary row; ``results.json`` collects the
rows, and every run is one HUD job. For ``libero``, seeds are the task's init states.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hud.agents.base import Agent
from hud.agents.types import AgentStep, ToolStep
from hud.eval import Job, LocalRuntime, rollout
from hud.settings import settings
from hud.telemetry.context import get_current_trace_id
from hud.types import MCPToolCall, MCPToolResult
from mcp.types import ImageContent, TextContent

from tabletop.sim import Task, layout

if TYPE_CHECKING:
    from hud.eval.run import Run

SYSTEM = (
    "You control a real robot arm through the `control` tools. Every call returns the "
    "camera frames and the arm's state. Work until the task is done, then stop calling tools "
    "and say so."
)


class ClaudeCode(Agent):
    """``claude -p`` as the agent, connected only to the env's ``control`` MCP server."""

    def __init__(self, model: str, out: Path, max_calls: int, max_usd: float) -> None:
        super().__init__()
        self.model = model
        self.out = out
        self.max_calls = max_calls
        self.max_usd = max_usd
        self.stream: Path | None = None

    async def __call__(self, run: Run) -> None:
        assert self.stream is not None
        config = self.stream.with_suffix(".mcp.json")
        url = run.client.binding("control").url
        # The trace-id header tells DirectControl which trace to stream state and video to.
        trace_id = get_current_trace_id()  # set for the rollout; run.trace gets it only at exit
        assert trace_id is not None
        server = {"type": "http", "url": url, "headers": {"trace-id": trace_id}}
        config.write_text(json.dumps({"mcpServers": {"control": server}}))
        with self.stream.open("w") as stream:
            proc = await asyncio.create_subprocess_exec(
                "claude",
                "-p",
                run.prompt or "",
                "--model",
                self.model,
                "--tools",
                "",
                "--strict-mcp-config",
                "--mcp-config",
                str(config),
                "--allowedTools",
                "mcp__control",
                "--append-system-prompt",
                SYSTEM,
                "--output-format",
                "stream-json",
                "--verbose",
                "--no-session-persistence",
                "--disable-slash-commands",
                "--max-budget-usd",
                str(self.max_usd),
                cwd=self.out,
                stdin=asyncio.subprocess.DEVNULL,
                # Straight to the file: a tool result carries base64 frames, so one stream
                # line is far past asyncio's 64 KiB line limit.
                stdout=stream,
            )
            while proc.returncode is None:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(proc.wait(), timeout=0.5)
                if self.stream.read_text().count('"type":"tool_result"') >= self.max_calls:
                    proc.terminate()
                    await proc.wait()
        record_steps(run, self.stream, self.model)
        run.trace.content = summarize(self.stream, None)["answer"]


def record_steps(run: Run, stream: Path, model: str) -> None:
    """Put the model's turns and tool round-trips from its stream on the trace.

    The stream emits each content block of a turn as its own event, all sharing the turn's
    message id. Recording uploads a step at once, so turns are assembled first, then recorded
    in stream order.
    """
    events = [json.loads(line) for line in stream.read_text().splitlines()]
    turns: dict[str, AgentStep] = {}
    calls: dict[str, MCPToolCall] = {}
    for event in events:
        if event["type"] != "assistant":
            continue
        message = event["message"]
        turn = turns.setdefault(message["id"], AgentStep(model=model))
        for block in message["content"]:
            if block["type"] == "text":
                turn.content = "\n".join(filter(None, [turn.content, block["text"]]))
            elif block["type"] == "tool_use":
                name = block["name"].removeprefix("mcp__control__")
                call = MCPToolCall(id=block["id"], name=name, arguments=block["input"])
                calls[call.id] = call
                turn.tool_calls.append(call)
    recorded: set[str] = set()
    for event in events:
        if event["type"] == "assistant" and event["message"]["id"] not in recorded:
            recorded.add(event["message"]["id"])
            run.record(turns[event["message"]["id"]])
        elif event["type"] == "user" and isinstance(event["message"].get("content"), list):
            for block in event["message"]["content"]:
                if block.get("type") != "tool_result":
                    continue
                result = MCPToolResult(
                    content=_blocks(block["content"]),
                    isError=bool(block.get("is_error")),
                    call_id=block["tool_use_id"],
                )
                run.record(ToolStep(call=calls.get(block["tool_use_id"]), result=result))


def _blocks(content: str | list[dict[str, Any]]) -> list[TextContent | ImageContent]:
    """A Claude tool result's content as MCP blocks (an error comes back as a plain string)."""
    if isinstance(content, str):
        return [TextContent(type="text", text=content)]
    return [
        TextContent(type="text", text=b["text"])
        if b["type"] == "text"
        else ImageContent(
            type="image", data=b["source"]["data"], mimeType=b["source"]["media_type"]
        )
        for b in content
    ]


def summarize(stream: Path, cube: list[float] | None, budget: float = 120) -> dict[str, Any]:
    """Tool calls, sim time, cost and the first move's distance from the cube."""
    calls: list[dict[str, Any]] = []
    remaining = None
    final: dict[str, Any] = {}
    for line in stream.read_text().splitlines():
        event = json.loads(line)
        content = event.get("message", {}).get("content", [])
        if event["type"] == "assistant":
            calls += [c for c in content if c.get("type") == "tool_use"]
        elif event["type"] == "user":
            for block in content:
                for item in (
                    block.get("content", []) if isinstance(block.get("content"), list) else []
                ):
                    if item.get("type") == "text" and (
                        m := re.search(r"Time remaining: ([\d.]+)", item["text"])
                    ):
                        remaining = float(m[1])
        elif event["type"] == "result":
            final = event
    first = next(
        (
            c["input"]["targets"]
            for c in calls
            if c["input"].get("targets", {}).get("x") is not None
        ),
        None,
    )
    miss = None
    if first is not None and cube is not None:
        miss = round(1000 * ((first["x"] - cube[0]) ** 2 + (first["y"] - cube[1]) ** 2) ** 0.5, 1)
    return {
        "calls": len(calls),
        "sim_s": None if remaining is None else round(budget - remaining, 1),
        "first_move_miss_mm": miss,
        "cost_usd": final.get("total_cost_usd"),
        "answer": final.get("result", ""),
    }


async def main(
    task: str,
    seeds: list[int],
    out: Path,
    model: str,
    max_calls: int,
    max_usd: float,
    task_id: int,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if task == "libero":
        # Seeds are LIBERO's fixed init states for one task of the suite.
        os.environ.setdefault("LIBERO_VIDEO", str(out.resolve()))
        from libero_env import env as module

        budget = module.MAX_STEPS[module.SUITE] / 20
        label = f"{module.SUITE} task {task_id} · {module.INTERFACE}"
    else:
        os.environ.setdefault("TABLETOP_VIDEO", str(out.resolve()))
        from tabletop import env as module

        budget = 120
        label = f"{task} · geometry {os.environ.get('TABLETOP_GEOMETRY', 'live')}"

    agent = ClaudeCode(model, out.resolve(), max_calls, max_usd)
    # One HUD job per benchmark, so its episodes' traces sit together on the platform.
    job = await Job.start(f"{label} · {model}", group=len(seeds))
    print(f"job: {settings.hud_web_url}/jobs/{uuid.UUID(job.id)}", flush=True)
    rows = []
    for seed in seeds:
        agent.stream = out.resolve() / f"{task}-seed{seed}.jsonl"
        if task == "libero":
            episode, cube = module.libero(task_id=task_id, init_state=seed), None
        else:
            episode, cube = getattr(module, task)(seed=seed), first_cube(task, seed)
        run = await rollout(episode, agent, runtime=LocalRuntime(module.env), job_id=job.id)
        row = {"seed": seed, "reward": run.reward, **summarize(agent.stream, cube, budget)}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "answer"}), flush=True)
        (out / "results.json").write_text(json.dumps(rows, indent=2))


def first_cube(task: str, seed: int) -> list[float]:
    """Where the first cube the prompt names starts."""
    episode = layout(cast("Task", task), seed)
    return episode.cubes[episode.pairs[0][0]].tolist()


def seed_range(text: str) -> list[int]:
    low, _, high = text.partition("-")
    return list(range(int(low), int(high or low) + 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["lift", "place", "place_both", "libero"])
    parser.add_argument("--task-id", type=int, default=0, help="libero: the suite's task")
    parser.add_argument("--seeds", type=seed_range, default=seed_range("0-4"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="claude-haiku-5-5")
    parser.add_argument("--max-calls", type=int, default=30)
    parser.add_argument("--max-usd", type=float, default=0.10)
    args = parser.parse_args()
    asyncio.run(
        main(
            args.task,
            args.seeds,
            args.out,
            args.model,
            args.max_calls,
            args.max_usd,
            args.task_id,
        )
    )
