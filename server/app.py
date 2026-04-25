"""
server/app.py — NegotiArena OpenEnv-Compliant Server with Frontend
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Any, Optional
from negotiarena_env import NegotiArenaEnv

app = FastAPI(title="NegotiArena", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_env: Optional[NegotiArenaEnv] = None
_episode_log: list[dict] = []

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    difficulty: str = "medium"

class StepRequest(BaseModel):
    agent_id: str
    action: dict[str, Any]

@app.get("/health")
def health():
    return {"status": "ok", "env": "NegotiArena", "version": "1.0.0"}

@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global _env, _episode_log
    _env = NegotiArenaEnv(seed=req.seed, difficulty=req.difficulty)
    observations = _env.reset()
    _episode_log = []
    state = _env.state()
    return {"episode_id": state.get("episode_id"), "observations": observations, "info": {"difficulty": req.difficulty}}

@app.post("/step")
def step(req: StepRequest):
    if _env is None:
        raise HTTPException(400, "Call /reset first")
    observations, rewards, done, info = _env.step(req.agent_id, req.action)
    _episode_log.append({"agent_id": req.agent_id, "action": req.action, "reward": rewards, "done": done, "turn": info.get("turn")})
    return {"observations": observations, "rewards": rewards, "done": done, "info": info}

@app.get("/state")
def state():
    if _env is None:
        raise HTTPException(400, "Call /reset first")
    return _env.state()

@app.get("/episode_log")
def episode_log():
    return {"steps": _episode_log, "total": len(_episode_log)}

@app.get("/", response_class=HTMLResponse)
def frontend():
    return open(os.path.join(os.path.dirname(__file__), "index.html"), encoding="utf-8").read()

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()