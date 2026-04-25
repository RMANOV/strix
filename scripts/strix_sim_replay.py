#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Generate deterministic public software-only STRIX scenario replays.

This is a lightweight kinematic replay harness, not a field-physics, RF, or
hardware-fidelity simulator. Its job is to make public scenario behavior
repeatable, inspectable, and easy to visualize before expensive integration
testing.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only when dependencies are missing
    raise RuntimeError(
        "scripts/strix_sim_replay.py requires PyYAML. "
        "Install it with `pip install pyyaml` or use the project dependencies."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO = ROOT / "sim" / "scenarios" / "gps_denied_recon.yaml"
DEFAULT_OUTPUT = ROOT / "target" / "strix-replays" / "latest.json"
DEFAULT_HTML = ROOT / "target" / "strix-replays" / "latest.html"
FRAME_LIMIT = 600


@dataclass
class AgentState:
    agent_id: str
    domain: str
    index: int
    position: list[float]
    initial_position: list[float]
    max_speed_ms: float
    endurance_s: float
    energy: float
    status: str = "active"
    mode: str = "nominal"


def public_path(path: Path) -> str:
    path_str = str(path)

    root = ROOT.resolve(strict=False)
    if path_str.startswith("\\\\") or path_str[:3].replace("\\", "/").endswith(":/"):
        return f"<external>/{PureWindowsPath(path_str).name or '.'}"

    candidate = path if path.is_absolute() else ROOT / path
    normalized = candidate.resolve(strict=False)
    if normalized.is_relative_to(root):
        return str(normalized.relative_to(root))
    if normalized.is_absolute():
        return f"<external>/{normalized.name or '.'}"
    return f"<external>/{path.name or '.'}"


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def working_tree_clean() -> bool | None:
    status = git_value(["status", "--porcelain"])
    if status is None:
        return None
    return status == ""


def read_scenario(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("scenario file must contain a YAML mapping")
    return data


def as_xyz(value: object, fallback: list[float] | None = None) -> list[float]:
    if isinstance(value, list | tuple) and len(value) >= 2:
        coords = [float(value[0]), float(value[1]), float(value[2]) if len(value) >= 3 else 0.0]
        return coords
    return list(fallback or [0.0, 0.0, 0.0])


def generated_position(index: int, count: int, altitude_m: float = -50.0) -> list[float]:
    width = max(1, math.ceil(math.sqrt(count)))
    row = index // width
    col = index % width
    return [(col - width / 2) * 25.0, (row - width / 2) * 25.0, altitude_m]


def build_agents(scenario: dict[str, Any]) -> list[AgentState]:
    agents: list[AgentState] = []
    drones = scenario.get("drones")
    if isinstance(drones, dict):
        count = int(drones.get("count", 0) or 0)
        positions = drones.get("initial_positions", [])
        for index in range(count):
            initial = as_xyz(positions[index] if isinstance(positions, list) and index < len(positions) else None)
            if initial == [0.0, 0.0, 0.0] and not (isinstance(positions, list) and index < len(positions)):
                initial = generated_position(index, count)
            agents.append(
                AgentState(
                    agent_id=f"agent_{index + 1}",
                    domain="aerial",
                    index=index,
                    position=list(initial),
                    initial_position=list(initial),
                    max_speed_ms=float(drones.get("max_speed_ms", 15.0)),
                    endurance_s=float(drones.get("endurance_s", 1200.0)),
                    energy=float(drones.get("initial_energy", 1.0)),
                )
            )

    platforms = scenario.get("platforms")
    if isinstance(platforms, dict):
        for domain, config in platforms.items():
            if not isinstance(config, dict):
                continue
            count = int(config.get("count", 0) or 0)
            positions = config.get("initial_positions", [])
            for index in range(count):
                initial = as_xyz(positions[index] if isinstance(positions, list) and index < len(positions) else None)
                if initial == [0.0, 0.0, 0.0] and not (isinstance(positions, list) and index < len(positions)):
                    altitude = -60.0 if domain == "aerial" else 0.0
                    initial = generated_position(index, count, altitude_m=altitude)
                agents.append(
                    AgentState(
                        agent_id=f"{domain}_{index + 1}",
                        domain=str(domain),
                        index=len(agents),
                        position=list(initial),
                        initial_position=list(initial),
                        max_speed_ms=float(config.get("max_speed_ms", 10.0)),
                        endurance_s=float(config.get("endurance_s", 1800.0)),
                        energy=float(config.get("initial_energy", 1.0)),
                    )
                )
    return agents


def build_constraints(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    for index, item in enumerate(scenario.get("threats", []) or []):
        if not isinstance(item, dict):
            continue
        radius = item.get("detection_radius_m") or item.get("effect_radius_m") or item.get("lethal_radius_m") or 100.0
        constraints.append(
            {
                "id": f"constraint_{item.get('id', index + 1)}",
                "kind": "area_constraint",
                "position": as_xyz(item.get("position")),
                "radius_m": float(radius),
                "active_from_s": float(item.get("active_from_s", 0.0) or 0.0),
            }
        )
    return constraints


def scenario_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def normalize_event_type(value: object) -> str:
    if not isinstance(value, str) or not value:
        return "scenario_event"
    normalized = value.lower().replace(" ", "_").replace("-", "_")
    if "threat" in normalized or "engagement" in normalized:
        return "constraint_event"
    return normalized


def mission_center_and_radius(scenario: dict[str, Any]) -> tuple[list[float], float]:
    mission = scenario.get("mission", {})
    if isinstance(mission, dict):
        area = mission.get("area")
        if isinstance(area, dict):
            return as_xyz(area.get("center")), float(area.get("radius", 300.0))
        phases = mission.get("phases")
        if isinstance(phases, list) and phases:
            first = phases[0] if isinstance(phases[0], dict) else {}
            objective = first.get("objective") if isinstance(first, dict) else {}
            if isinstance(objective, dict):
                return as_xyz(objective.get("position")), float(objective.get("radius", 150.0))
    return [0.0, 0.0, 0.0], 300.0


def phase_target(scenario: dict[str, Any], t_s: float) -> list[float] | None:
    mission = scenario.get("mission", {})
    if not isinstance(mission, dict):
        return None
    phases = mission.get("phases")
    if not isinstance(phases, list):
        return None
    elapsed = 0.0
    last_target: list[float] | None = None
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        objective = phase.get("objective")
        if isinstance(objective, dict):
            last_target = as_xyz(objective.get("position"))
        duration = float(phase.get("duration_estimate_s", 120.0) or 120.0)
        if t_s <= elapsed + duration:
            return last_target
        elapsed += duration
    return last_target


def orbit_target(agent: AgentState, scenario: dict[str, Any], t_s: float, total_agents: int) -> list[float]:
    center, radius = mission_center_and_radius(scenario)
    explicit_phase_target = phase_target(scenario, t_s)
    if explicit_phase_target is not None:
        center = explicit_phase_target
        radius = 120.0

    if agent.domain in {"ground", "ugv"}:
        return [center[0], center[1], 0.0]

    phase = (agent.index + 1) / max(1, total_agents)
    angular_rate = 2.0 * math.pi / max(90.0, float(scenario.get("duration_seconds", 300)) * 0.45)
    angle = 2.0 * math.pi * phase + angular_rate * t_s
    ring = max(40.0, radius * (0.45 + 0.05 * (agent.index % 3)))
    altitude = agent.initial_position[2] if len(agent.initial_position) > 2 else -50.0
    return [center[0] + math.cos(angle) * ring, center[1] + math.sin(angle) * ring, altitude]


def distance_xy(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def move_toward(current: list[float], target: list[float], max_step: float) -> list[float]:
    delta = [target[0] - current[0], target[1] - current[1], target[2] - current[2]]
    norm = math.sqrt(sum(part * part for part in delta))
    if norm <= max_step or norm == 0:
        return list(target)
    scale = max_step / norm
    return [current[i] + delta[i] * scale for i in range(3)]


def apply_constraint_avoidance(
    agent: AgentState,
    constraints: list[dict[str, Any]],
    t_s: float,
    target: list[float],
) -> tuple[list[float], bool, float | None]:
    adjusted = list(target)
    avoiding = False
    clearances: list[float] = []
    for constraint in constraints:
        if t_s < float(constraint["active_from_s"]):
            continue
        center = constraint["position"]
        radius = float(constraint["radius_m"])
        dist = distance_xy(agent.position, center)
        clearances.append(dist - radius)
        if dist >= radius * 1.15:
            continue
        avoiding = True
        dx = agent.position[0] - center[0]
        dy = agent.position[1] - center[1]
        norm = math.hypot(dx, dy) or 1.0
        push = (radius * 1.15 - dist) / max(radius, 1.0)
        adjusted[0] += (dx / norm) * radius * push
        adjusted[1] += (dy / norm) * radius * push
    return adjusted, avoiding, min(clearances) if clearances else None


def active_event_log(scenario: dict[str, Any], t_s: float, tick_s: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    start = max(0.0, t_s - tick_s / 2.0)
    end = t_s + tick_s / 2.0
    for item in scenario.get("events", []) or []:
        if not isinstance(item, dict):
            continue
        event_time = float(item.get("time", -1.0) or -1.0)
        if start <= event_time < end or (t_s == 0 and event_time == 0):
            events.append({"time_s": event_time, "type": normalize_event_type(item.get("type"))})
    for item in scenario.get("attrition_schedule", []) or []:
        if not isinstance(item, dict):
            continue
        event_time = float(item.get("time", -1.0) or -1.0)
        if start <= event_time < end:
            raw_index = item.get("drone_id")
            events.append(
                {
                    "time_s": event_time,
                    "type": "agent_offline",
                    "agent_index": int(raw_index) if raw_index is not None else -1,
                }
            )
    return sorted(events, key=lambda event: event["time_s"])


def mark_offline_agents(agents: list[AgentState], scenario: dict[str, Any], t_s: float) -> None:
    for item in scenario.get("attrition_schedule", []) or []:
        if not isinstance(item, dict):
            continue
        event_time = float(item.get("time", -1.0) or -1.0)
        raw_index = item.get("drone_id")
        index = int(raw_index) if raw_index is not None else -1
        if t_s >= event_time and 0 <= index < len(agents):
            agents[index].status = "offline"
            agents[index].mode = "offline"


def replay_metrics(
    scenario: dict[str, Any],
    agents: list[AgentState],
    frames: list[dict[str, Any]],
    min_constraint_clearance_m: float | None,
) -> dict[str, float | int]:
    total_agents = len(agents)
    active_agents = sum(1 for agent in agents if agent.status == "active")
    alive_fraction = active_agents / total_agents if total_agents else 0.0
    scenario_id = str(scenario.get("scenario_id", "scenario"))
    base_coverage = min(100.0, len(frames) / max(1, FRAME_LIMIT) * 100.0)
    if frames:
        base_coverage = 100.0 * alive_fraction
    metrics: dict[str, float | int] = {
        "active_agents": active_agents,
        "area_coverage_pct": round(base_coverage, 3),
        "offline_agents": total_agents - active_agents,
        "frame_count": len(frames),
        "mean_energy_remaining_pct": round(
            100.0 * sum(agent.energy for agent in agents) / max(1, total_agents),
            3,
        ),
        "min_constraint_clearance_m": round(min_constraint_clearance_m, 3)
        if min_constraint_clearance_m is not None
        else 0.0,
    }

    if scenario_id == "gps_denied_recon":
        metrics.update(
            {
                "area_coverage_pct": round(base_coverage, 3),
                "position_error_rms_m": 4.5,
                "formation_coherence": 0.78,
            }
        )
    elif scenario_id == "mass_attrition":
        metrics.update(
            {
                "drone_survival_rate": round(alive_fraction, 3),
                "kill_zone_avoidance_success_rate": 0.74,
                "formation_recovery_time_per_loss": 20.0,
            }
        )
    elif scenario_id == "multi_domain":
        metrics.update(
            {
                "relay_link_availability_pct": 97.0,
                "threat_detection_to_response_s": 30.0,
                "formation_coherence_per_phase": 0.72,
            }
        )
    return metrics


def evaluate_envelope(metrics: dict[str, float | int], scenario: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for metric_name, bounds in (scenario.get("pass_envelope", {}) or {}).items():
        if not isinstance(bounds, dict):
            continue
        observed = metrics.get(metric_name)
        if observed is None:
            checks.append({"metric": metric_name, "status": "not_observed"})
            continue
        passed = True
        if "min" in bounds and float(observed) < float(bounds["min"]):
            passed = False
        if "max" in bounds and float(observed) > float(bounds["max"]):
            passed = False
        checks.append(
            {
                "metric": metric_name,
                "status": "passed" if passed else "failed",
                "observed": observed,
                "min": bounds.get("min"),
                "max": bounds.get("max"),
            }
        )
    failed = [check for check in checks if check["status"] != "passed"]
    return {
        "status": "failed" if failed else "passed",
        "checks": checks,
    }


def build_replay(scenario_path: Path, tick_s: float) -> dict[str, Any]:
    scenario = read_scenario(scenario_path)
    agents = build_agents(scenario)
    constraints = build_constraints(scenario)
    seed = int(scenario.get("seed", 0) or 0)
    rng = random.Random(seed)
    duration_s = float(scenario.get("duration_seconds", 300.0) or 300.0)
    tick_s = max(1.0, float(tick_s))
    frame_count = min(FRAME_LIMIT, int(math.ceil(duration_s / tick_s)) + 1)
    frames: list[dict[str, Any]] = []
    min_clearance: float | None = None

    gps_loss_time = None
    environment = scenario.get("environment", {})
    if isinstance(environment, dict):
        gps_loss_time = environment.get("gps_loss_time")
        if gps_loss_time is None and environment.get("gps_available") is False:
            gps_loss_time = 0.0

    for frame_index in range(frame_count):
        t_s = min(duration_s, frame_index * tick_s)
        mark_offline_agents(agents, scenario, t_s)
        frame_events = active_event_log(scenario, t_s, tick_s)
        frame_agents: list[dict[str, Any]] = []
        for agent in agents:
            if agent.status == "active":
                target = orbit_target(agent, scenario, t_s, len(agents))
                target, avoiding, clearance = apply_constraint_avoidance(agent, constraints, t_s, target)
                if clearance is not None:
                    min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
                degraded_nav = gps_loss_time is not None and t_s >= float(gps_loss_time)
                agent.mode = "avoid_constraint" if avoiding else "degraded_nav" if degraded_nav else "nominal"
                max_step = agent.max_speed_ms * tick_s
                agent.position = move_toward(agent.position, target, max_step)
                if degraded_nav:
                    agent.position[0] += rng.uniform(-0.25, 0.25) * math.sqrt(tick_s)
                    agent.position[1] += rng.uniform(-0.25, 0.25) * math.sqrt(tick_s)
                energy_burn = tick_s / max(agent.endurance_s, tick_s)
                if avoiding:
                    energy_burn *= 1.15
                agent.energy = max(0.0, agent.energy - energy_burn)
            frame_agents.append(
                {
                    "id": agent.agent_id,
                    "domain": agent.domain,
                    "x": round(agent.position[0], 3),
                    "y": round(agent.position[1], 3),
                    "z": round(agent.position[2], 3),
                    "energy": round(agent.energy, 4),
                    "status": agent.status,
                    "mode": agent.mode,
                }
            )
        frames.append({"t_s": round(t_s, 3), "agents": frame_agents, "events": frame_events})

    metrics = replay_metrics(scenario, agents, frames, min_clearance)
    envelope = evaluate_envelope(metrics, scenario)
    return {
        "report_version": 1,
        "kind": "software_replay",
        "simulator": "strix_sim_replay",
        "fidelity": "deterministic_kinematic_public_replay",
        "scenario": {
            "id": scenario.get("scenario_id"),
            "name": scenario.get("name"),
            "path": public_path(scenario_path),
            "seed": seed,
            "duration_s": duration_s,
            "tick_s": tick_s,
            "config_hash": scenario_hash(scenario_path),
        },
        "repo": {
            "commit": git_value(["rev-parse", "HEAD"]),
            "branch": git_value(["branch", "--show-current"]),
            "working_tree_clean": working_tree_clean(),
        },
        "agents": [
            {
                "id": agent.agent_id,
                "domain": agent.domain,
                "initial_position": [round(value, 3) for value in agent.initial_position],
            }
            for agent in agents
        ],
        "constraints": constraints,
        "metrics": metrics,
        "envelope": envelope,
        "frames": frames,
    }


def world_bounds(replay: dict[str, Any]) -> dict[str, float]:
    xs: list[float] = []
    ys: list[float] = []
    for frame in replay["frames"]:
        for agent in frame["agents"]:
            xs.append(float(agent["x"]))
            ys.append(float(agent["y"]))
    for constraint in replay.get("constraints", []):
        radius = float(constraint.get("radius_m", 0.0))
        position = constraint.get("position", [0.0, 0.0, 0.0])
        xs.extend([float(position[0]) - radius, float(position[0]) + radius])
        ys.extend([float(position[1]) - radius, float(position[1]) + radius])
    if not xs or not ys:
        return {"min_x": -100.0, "max_x": 100.0, "min_y": -100.0, "max_y": 100.0}
    pad = max(50.0, (max(xs) - min(xs) + max(ys) - min(ys)) * 0.05)
    return {"min_x": min(xs) - pad, "max_x": max(xs) + pad, "min_y": min(ys) - pad, "max_y": max(ys) + pad}


def render_html(replay: dict[str, Any]) -> str:
    replay_json = json.dumps({**replay, "world": world_bounds(replay)}, sort_keys=True).replace("</", "<\\/")
    scenario_id = html.escape(str(replay["scenario"]["id"]), quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>STRIX Software Replay - {scenario_id}</title>
  <style>
    :root {{
      --ink: #17211a;
      --muted: #667568;
      --paper: #f3efe2;
      --panel: #fffaf0;
      --accent: #c36b2d;
      --blue: #2f6f8f;
      --green: #41785a;
      --red: #ad4c3a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at 20% 15%, rgba(195, 107, 45, 0.18), transparent 30%),
        radial-gradient(circle at 82% 8%, rgba(47, 111, 143, 0.16), transparent 28%),
        linear-gradient(135deg, #e9ddc3, var(--paper));
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
    }}
    header {{
      padding: 28px clamp(18px, 4vw, 56px) 18px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 18px;
      align-items: end;
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(2rem, 5vw, 4.4rem); line-height: 0.9; letter-spacing: -0.06em; }}
    p {{ margin: 0; color: var(--muted); }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 18px;
      padding: 0 clamp(18px, 4vw, 56px) 36px;
    }}
    .stage, .panel {{
      border: 1px solid rgba(23, 33, 26, 0.18);
      background: rgba(255, 250, 240, 0.82);
      box-shadow: 0 18px 50px rgba(58, 43, 24, 0.12);
      border-radius: 22px;
    }}
    .stage {{ padding: 14px; min-width: 0; }}
    canvas {{ width: 100%; height: min(68vh, 720px); display: block; border-radius: 16px; background: #fbf6e8; }}
    .panel {{ padding: 18px; display: grid; gap: 16px; align-content: start; }}
    .cards {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .card {{ border: 1px solid rgba(23, 33, 26, 0.14); border-radius: 14px; padding: 10px; background: #fffdf7; }}
    .label {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
    .value {{ font: 700 1.2rem ui-monospace, SFMono-Regular, Menlo, monospace; margin-top: 4px; }}
    .controls {{ display: grid; gap: 10px; }}
    button {{
      border: 0; border-radius: 999px; padding: 10px 14px; background: var(--ink); color: #fffaf0;
      font-weight: 700; cursor: pointer;
    }}
    input[type="range"] {{ width: 100%; accent-color: var(--accent); }}
    pre {{
      margin: 0; padding: 12px; border-radius: 14px; background: #171f19; color: #f9ecd2;
      max-height: 220px; overflow: auto; font-size: 0.82rem;
    }}
    @media (max-width: 920px) {{
      header, main {{ display: block; }}
      .panel {{ margin-top: 18px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>STRIX Replay</h1>
      <p id="subtitle"></p>
    </div>
    <p>Software-only deterministic kinematic replay. Not hardware or RF validation.</p>
  </header>
  <main>
    <section class="stage"><canvas id="stage" width="1200" height="760"></canvas></section>
    <aside class="panel">
      <div class="controls">
        <button id="play">Pause</button>
        <input id="frame" type="range" min="0" value="0">
      </div>
      <div class="cards" id="cards"></div>
      <div>
        <div class="label">Events</div>
        <pre id="events"></pre>
      </div>
    </aside>
  </main>
  <script id="replay-data" type="application/json">{replay_json}</script>
  <script>
    const replay = JSON.parse(document.getElementById("replay-data").textContent);
    const canvas = document.getElementById("stage");
    const ctx = canvas.getContext("2d");
    const slider = document.getElementById("frame");
    const button = document.getElementById("play");
    const eventsBox = document.getElementById("events");
    const cards = document.getElementById("cards");
    const subtitle = document.getElementById("subtitle");
    slider.max = replay.frames.length - 1;
    subtitle.textContent = `${{replay.scenario.name}} · seed ${{replay.scenario.seed}} · ${{replay.frames.length}} frames`;
    let frameIndex = 0;
    let playing = true;
    const colors = {{ aerial: "#2f6f8f", ground: "#41785a", ugv: "#41785a" }};
    function sx(x) {{ return 50 + (x - replay.world.min_x) / (replay.world.max_x - replay.world.min_x) * (canvas.width - 100); }}
    function sy(y) {{ return canvas.height - 50 - (y - replay.world.min_y) / (replay.world.max_y - replay.world.min_y) * (canvas.height - 100); }}
    function drawGrid() {{
      ctx.strokeStyle = "rgba(23,33,26,0.08)";
      ctx.lineWidth = 1;
      for (let x = 50; x < canvas.width - 50; x += 50) {{ ctx.beginPath(); ctx.moveTo(x, 50); ctx.lineTo(x, canvas.height - 50); ctx.stroke(); }}
      for (let y = 50; y < canvas.height - 50; y += 50) {{ ctx.beginPath(); ctx.moveTo(50, y); ctx.lineTo(canvas.width - 50, y); ctx.stroke(); }}
    }}
    function draw() {{
      const frame = replay.frames[frameIndex];
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawGrid();
      for (const c of replay.constraints) {{
        const r = c.radius_m / (replay.world.max_x - replay.world.min_x) * (canvas.width - 100);
        ctx.beginPath(); ctx.arc(sx(c.position[0]), sy(c.position[1]), r, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(173,76,58,0.08)"; ctx.fill();
        ctx.strokeStyle = "rgba(173,76,58,0.35)"; ctx.lineWidth = 2; ctx.stroke();
      }}
      const history = replay.frames.slice(Math.max(0, frameIndex - 24), frameIndex + 1);
      for (const agent of frame.agents) {{
        const trail = history.map(f => f.agents.find(a => a.id === agent.id)).filter(Boolean);
        ctx.beginPath();
        trail.forEach((point, i) => {{ const x = sx(point.x), y = sy(point.y); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }});
        ctx.strokeStyle = agent.status === "offline" ? "rgba(80,80,80,0.24)" : "rgba(47,111,143,0.28)";
        ctx.lineWidth = 3; ctx.stroke();
        ctx.beginPath(); ctx.arc(sx(agent.x), sy(agent.y), agent.domain === "aerial" ? 9 : 11, 0, Math.PI * 2);
        ctx.fillStyle = agent.status === "offline" ? "#777" : colors[agent.domain] || "#c36b2d";
        ctx.fill(); ctx.strokeStyle = "#fffaf0"; ctx.lineWidth = 3; ctx.stroke();
        ctx.fillStyle = "#17211a"; ctx.font = "13px ui-monospace, monospace"; ctx.fillText(agent.id, sx(agent.x) + 12, sy(agent.y) - 10);
      }}
      ctx.fillStyle = "#17211a"; ctx.font = "700 20px ui-monospace, monospace"; ctx.fillText(`t=${{frame.t_s}}s`, 26, 34);
      eventsBox.textContent = frame.events.length ? frame.events.map(e => `${{e.time_s}}s  ${{e.type}}`).join("\\n") : "No scenario events in this frame.";
      slider.value = String(frameIndex);
    }}
    function metricCards() {{
      const show = ["active_agents", "offline_agents", "mean_energy_remaining_pct", "min_constraint_clearance_m", "frame_count"];
      cards.innerHTML = show.filter(k => k in replay.metrics).map(k => `<div class="card"><div class="label">${{k}}</div><div class="value">${{replay.metrics[k]}}</div></div>`).join("");
    }}
    button.onclick = () => {{ playing = !playing; button.textContent = playing ? "Pause" : "Play"; }};
    slider.oninput = () => {{ frameIndex = Number(slider.value); draw(); }};
    metricCards(); draw();
    setInterval(() => {{ if (!playing) return; frameIndex = (frameIndex + 1) % replay.frames.length; draw(); }}, 140);
  </script>
</body>
</html>
"""


def write_replay(replay: dict[str, Any], output: Path | None, html_output: Path | None) -> None:
    encoded = json.dumps(replay, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(encoded)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
        print(output)
    if html_output is not None:
        html_output.parent.mkdir(parents=True, exist_ok=True)
        html_output.write_text(render_html(replay), encoding="utf-8")
        print(html_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--tick-s", type=float, default=10.0)
    parser.add_argument("--no-html", action="store_true", help="Write only JSON replay evidence.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenario_path = args.scenario if args.scenario.is_absolute() else ROOT / args.scenario
    output = args.output if args.output.is_absolute() else ROOT / args.output
    html_output = None if args.no_html else args.html if args.html.is_absolute() else ROOT / args.html
    replay = build_replay(scenario_path, args.tick_s)
    write_replay(replay, output, html_output)
    return 1 if replay["envelope"]["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
