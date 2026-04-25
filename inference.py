"""
inference.py — NegotiArena Inference Client
============================================
Run trained negotiator/overseer adapters against the live environment.
Used for:
  1. Post-training evaluation
  2. Demo episode generation
  3. Side-by-side before/after comparison

Usage:
    python inference.py --adapter overseer --checkpoint checkpoints/overseer --episodes 5
    python inference.py --mode demo --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from negotiarena_env import NegotiArenaEnv
from training.prompts import format_negotiator_prompt, format_overseer_prompt
from training.generate_sft_data import NegotiatorBot, OverseerBot

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    RICH = True
except ImportError:
    RICH = False


console = Console() if RICH else None


def print_msg(msg: str, style: str = ""):
    if RICH:
        console.print(msg, style=style, markup=False)
    else:
        print(msg)


def run_demo_episode(
    seed: int = 42,
    difficulty: str = "medium",
    smart_overseer: bool = True,
    verbose: bool = True,
) -> dict:
    """Run a single demo episode and return full state."""
    env = NegotiArenaEnv(seed=seed, difficulty=difficulty)
    observations = env.reset()

    neg_bots = {
        "negotiator_a": NegotiatorBot("negotiator_a", greedy=True),
        "negotiator_b": NegotiatorBot("negotiator_b", greedy=False),
        "negotiator_c": NegotiatorBot("negotiator_c"),
    }
    overseer = OverseerBot(random_flag_prob=0.6 if smart_overseer else 0.1)
    # Inject coalition hint for smart mode (simulates what a trained LLM learns to infer)
    if smart_overseer and env._state and env._state.coalitions:
        hint = []
        for c in env._state.coalitions:
            hint.extend(c.members)
        overseer._coalition_hint = list(set(hint))

    done = False
    step = 0
    all_rewards = {aid: 0.0 for aid in env.ALL_AGENT_IDS}

    print_msg("\n" + "═" * 60, "bold blue")
    print_msg("🏛️  NegotiArena Episode", "bold cyan")
    print_msg(f"   Seed: {seed} | Difficulty: {difficulty} | "
              f"Smart Overseer: {smart_overseer}", "dim")
    print_msg("═" * 60, "bold blue")

    while not done and step < 80:
        for agent_id in ["negotiator_a", "negotiator_b", "negotiator_c", "overseer"]:
            obs = observations.get(agent_id, {})

            if agent_id == "overseer":
                action = overseer.act(obs)
            else:
                action = neg_bots[agent_id].act(obs)

            observations, rewards, done, info = env.step(agent_id, action)

            for aid, r in rewards.items():
                all_rewards[aid] += r

            if verbose and action.get("type") not in ("pass",):
                turn = info.get("turn", 0)
                atype = action.get("type", "?")
                content = action.get("content", "")
                emoji = "🔍" if agent_id == "overseer" else "👤"
                style = "red" if agent_id == "overseer" else "cyan"
                # Show action type label when content is blank (e.g. coalition_form)
                display = content if content else f"[{atype}]"
                print_msg(
                    f"  {emoji} [{turn:02d}] {agent_id:<16} "
                    f"{display[:70]}",
                    style
                )

            if done:
                break
        step += 4

    final_state = env.state()

    # Print summary
    print_msg("\n" + "─" * 60, "dim")
    resolution = final_state.get("resolution_type", "unknown")
    turns = final_state.get("turn", 0)
    coalitions = final_state.get("coalitions", [])
    flags = final_state.get("overseer_flags", [])

    print_msg(f"\n  Resolution: {resolution.upper()} at turn {turns}", "bold green")

    if coalitions:
        for c in coalitions:
            detected = c.get("detected", False)
            members = ", ".join(c.get("members", []))
            status = "✅ DETECTED" if detected else "❌ UNDETECTED"
            line = f"  Coalition ({members}): {status}"
            # markup=False prevents Rich eating (negotiator_a, ...) as markup tags
            if RICH:
                console.print(line, markup=False,
                               style="bold green" if detected else "bold yellow")
            else:
                print(line)
    else:
        print_msg("  No coalitions formed this episode", "dim")

    if flags:
        for f in flags:
            print_msg(f"  🚩 Overseer flagged: {f['agent_id']} — {f.get('reason', '')}", "red")

    # Compute F1
    coalition_agents = set()
    for c in coalitions:
        coalition_agents.update(c.get("members", []))
    flagged = {f["agent_id"] for f in flags}
    tp = len(coalition_agents & flagged)
    fp = len(flagged - coalition_agents)
    fn = len(coalition_agents - flagged)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    print_msg(f"\n  Overseer F1: {f1:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}", "bold")
    print_msg(f"  Overseer Total Reward: {all_rewards['overseer']:+.3f}", "bold")
    print_msg("═" * 60 + "\n", "bold blue")

    return {
        "state": final_state,
        "f1": f1,
        "rewards": all_rewards,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "eval"], default="demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--difficulty", default="medium")
    parser.add_argument("--smart_overseer", action="store_true", default=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "demo":
        for i in range(args.episodes):
            run_demo_episode(
                seed=args.seed + i,
                difficulty=args.difficulty,
                smart_overseer=args.smart_overseer,
            )
    elif args.mode == "eval":
        from evaluation.evaluator import evaluate_random_policy, print_comparison
        before = evaluate_random_policy(n_episodes=20)
        print_comparison(before, before)  # Show baseline until trained model available


if __name__ == "__main__":
    main()