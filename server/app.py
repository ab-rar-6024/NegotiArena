"""
server/app.py — NegotiArena FastAPI Server (OpenEnv-compatible)
===============================================================
Endpoints:
  POST /reset        — start new episode, returns initial observations
  POST /step         — advance one step, returns obs/rewards/done/info
  GET  /state        — full state dump (for eval/logging)
  GET  /health       — liveness probe
  GET  /metrics      — current episode metrics for demo dashboard
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Optional

from negotiarena_env import NegotiArenaEnv


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NegotiArena",
    description="Multi-agent negotiation environment for scalable oversight training",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global env instance (single-session for hackathon; scale with session IDs in prod)
_env: Optional[NegotiArenaEnv] = None
_episode_metrics: list[dict] = []


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    difficulty: str = "medium"   # easy | medium | hard


class StepRequest(BaseModel):
    agent_id: str
    action: dict[str, Any]


class ResetResponse(BaseModel):
    episode_id: str
    observations: dict[str, Any]
    message: str = "Episode started"


class StepResponse(BaseModel):
    observations: dict[str, Any]
    rewards: dict[str, float]
    done: bool
    info: dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "env_ready": _env is not None}


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    global _env, _episode_metrics
    _env = NegotiArenaEnv(seed=req.seed, difficulty=req.difficulty)
    observations = _env.reset()
    state = _env.state()
    # Log episode start metrics
    _episode_metrics = []
    return ResetResponse(
        episode_id=state.get("episode_id", "unknown"),
        observations=observations,
    )


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    observations, rewards, done, info = _env.step(req.agent_id, req.action)

    # Track metrics for demo dashboard
    _episode_metrics.append({
        "turn": info.get("turn"),
        "rewards": rewards,
        "done": done,
    })

    return StepResponse(
        observations=observations,
        rewards=rewards,
        done=done,
        info=info,
    )


@app.get("/state")
def state():
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return _env.state()


@app.get("/metrics")
def metrics():
    """Current episode metrics for live demo dashboard."""
    if not _episode_metrics:
        return {"turns": [], "overseer_rewards": [], "deal_quality": []}
    turns = [m["turn"] for m in _episode_metrics if m["turn"] is not None]
    overseer_rewards = [m["rewards"].get("overseer", 0.0) for m in _episode_metrics]
    avg_deal = [
        sum(m["rewards"].get(aid, 0.0) for aid in ["negotiator_a", "negotiator_b", "negotiator_c"]) / 3
        for m in _episode_metrics
    ]
    return {
        "turns": turns,
        "overseer_rewards": overseer_rewards,
        "avg_deal_quality": avg_deal,
        "total_episodes": len(_episode_metrics),
    }


@app.get("/")
def root():
    return {
        "name": "NegotiArena",
        "version": "1.0.0",
        "theme": "Multi-Agent Interactions + Fleet AI / Scalable Oversight",
        "endpoints": ["/reset", "/step", "/state", "/health", "/metrics"],
    }


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()