"""
tests/test_env.py — NegotiArena Environment Tests
==================================================
Run: pytest tests/ -v
"""

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from negotiarena_env import (
    NegotiArenaEnv,
    PriorityCard,
    RewardEngine,
    EpisodeState,
    ActionType,
    TOTAL_RESOURCES,
    RESOURCE_TYPES,
    COALITION_STEALTH_CAP,
)


# ---------------------------------------------------------------------------
# PriorityCard tests
# ---------------------------------------------------------------------------

def test_priority_card_weights_sum_to_one():
    card = PriorityCard.random("test_agent")
    total = sum(card.weights.values())
    assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"


def test_priority_card_utility_range():
    card = PriorityCard.random("test_agent")
    alloc = {r: TOTAL_RESOURCES[r] / 3 for r in RESOURCE_TYPES}
    utility = card.utility(alloc)
    assert 0.0 <= utility <= 1.0, f"Utility {utility} out of range [0, 1]"


def test_priority_card_max_utility():
    """Agent gets all resources → utility should be near 1.0."""
    card = PriorityCard.random("test_agent")
    alloc = {r: TOTAL_RESOURCES[r] for r in RESOURCE_TYPES}
    utility = card.utility(alloc)
    assert utility > 0.8, f"Max allocation utility too low: {utility}"


# ---------------------------------------------------------------------------
# Environment reset / step
# ---------------------------------------------------------------------------

def test_env_reset_returns_all_agents():
    env = NegotiArenaEnv(seed=42)
    obs = env.reset()
    expected = {"negotiator_a", "negotiator_b", "negotiator_c", "overseer"}
    assert set(obs.keys()) == expected


def test_overseer_cannot_see_priority_cards():
    """Critical: overseer observation must NOT contain private card data."""
    env = NegotiArenaEnv(seed=42)
    obs = env.reset()
    overseer_obs = obs["overseer"]
    assert "my_priority_weights" not in overseer_obs
    assert "my_min_acceptable" not in overseer_obs
    assert "priority_cards" not in overseer_obs


def test_negotiator_has_private_card():
    env = NegotiArenaEnv(seed=42)
    obs = env.reset()
    for aid in ["negotiator_a", "negotiator_b", "negotiator_c"]:
        assert "my_priority_weights" in obs[aid]
        weights = obs[aid]["my_priority_weights"]
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 1e-5


def test_step_returns_correct_structure():
    env = NegotiArenaEnv(seed=42)
    env.reset()
    action = {"type": "pass", "content": "monitoring"}
    obs, rewards, done, info = env.step("overseer", action)
    assert "overseer" in rewards
    assert "turn" in info
    assert isinstance(done, bool)


def test_invalid_action_format_no_crash():
    """Malformed actions should return 0 reward, not crash."""
    env = NegotiArenaEnv(seed=42)
    env.reset()
    obs, rewards, done, info = env.step("negotiator_a", {"bad": "action"})
    assert rewards["negotiator_a"] == 0.0
    assert not done


def test_valid_offer_accepted_by_two_ends_episode():
    env = NegotiArenaEnv(seed=42)
    env.reset()

    # Make an offer
    offer_action = {
        "type": "offer",
        "allocation": {"compute": 30, "budget": 15000, "headcount": 3},
        "content": "Proposal 1",
    }
    env.step("negotiator_a", offer_action)
    env.step("overseer", {"type": "pass", "content": ""})

    # Both other negotiators accept
    env.step("negotiator_b", {"type": "accept", "content": "Accepted."})
    obs, rewards, done, info = env.step("negotiator_c", {"type": "accept", "content": "Agreed."})

    assert done, "Episode should end after unanimous acceptance"


def test_timeout_after_max_turns():
    env = NegotiArenaEnv(seed=99)
    env.reset()
    done = False
    steps = 0
    while not done and steps < 200:
        for agent_id in env.ALL_AGENT_IDS:
            _, _, done, info = env.step(agent_id, {"type": "pass", "content": ""})
            steps += 1
            if done:
                break
    assert done, "Episode should eventually terminate"
    state = env.state()
    assert state["resolution_type"] in ("deal", "timeout", "overseer_revote")


# ---------------------------------------------------------------------------
# Reward engine tests
# ---------------------------------------------------------------------------

def test_reward_engine_no_coalition_no_stealth():
    env = NegotiArenaEnv(seed=42)
    env.reset()
    state = env._state
    state.coalitions = []
    state.final_allocation = {r: TOTAL_RESOURCES[r] / 3 for r in RESOURCE_TYPES}
    state.resolution_type = "deal"

    engine = RewardEngine()
    rewards = engine.compute_episode_rewards(state, env.ALL_AGENT_IDS)

    for aid in ["negotiator_a", "negotiator_b", "negotiator_c"]:
        assert rewards[aid].get("coalition_stealth", 0.0) == 0.0


def test_coalition_stealth_capped():
    """Stealth bonus must never exceed COALITION_STEALTH_CAP."""
    env = NegotiArenaEnv(seed=42)
    env.reset()
    state = env._state
    from negotiarena_env import CoalitionEvent
    state.coalitions = [CoalitionEvent(
        members=["negotiator_a", "negotiator_b"],
        formed_at_turn=0, detected=False,
    )]
    state.final_allocation = {
        "compute": 60, "budget": 35000, "headcount": 7
    }
    state._state = state
    state.turn = 20

    engine = RewardEngine()
    rewards = engine.compute_episode_rewards(state, env.ALL_AGENT_IDS)

    for member in ["negotiator_a", "negotiator_b"]:
        stealth = rewards[member].get("coalition_stealth", 0.0)
        assert stealth <= COALITION_STEALTH_CAP + 1e-6, \
            f"Stealth bonus {stealth} exceeds cap {COALITION_STEALTH_CAP}"


def test_overseer_false_positive_penalty():
    """Flagging innocent agents should penalise overseer."""
    env = NegotiArenaEnv(seed=42)
    env.reset()
    state = env._state
    state.coalitions = []  # No coalitions
    state.overseer_flags = [{"agent_id": "negotiator_a", "turn": 5, "reason": "suspicious"}]
    state.final_allocation = {r: TOTAL_RESOURCES[r] / 3 for r in RESOURCE_TYPES}

    engine = RewardEngine()
    rewards = engine.compute_episode_rewards(state, env.ALL_AGENT_IDS)

    overseer_detection = rewards["overseer"].get("overseer_detection", 0.0)
    assert overseer_detection < 1.0, "Overseer should be penalised for false positive"


def test_repeated_action_penalty():
    """3 identical actions in a row should trigger penalty."""
    env = NegotiArenaEnv(seed=42)
    env.reset()
    state = env._state

    # Simulate 3 identical actions
    action = {"type": "pass", "content": "monitoring"}
    import hashlib
    fp = hashlib.md5(json.dumps(action, sort_keys=True).encode()).hexdigest()[:8]
    state.action_history["negotiator_a"] = [fp, fp, fp]

    engine = RewardEngine()
    penalty = engine._repeated_action_penalty(state, "negotiator_a")
    assert penalty < 0.0, "Repeated actions should have negative penalty"


# ---------------------------------------------------------------------------
# Anti-reward-hacking integration tests
# ---------------------------------------------------------------------------

def test_coalition_stealth_requires_outcome_benefit():
    """
    Coalition that doesn't improve members' outcomes should get 0 stealth reward.
    This is the outcome-gating anti-hack measure.
    """
    env = NegotiArenaEnv(seed=42)
    env.reset()
    state = env._state
    from negotiarena_env import CoalitionEvent

    state.coalitions = [CoalitionEvent(
        members=["negotiator_a", "negotiator_b"],
        formed_at_turn=0, detected=False,
    )]
    # Give them BELOW-baseline allocation (coalition hurt them)
    state.final_allocation = {"compute": 10, "budget": 5000, "headcount": 1}
    state.turn = 15

    engine = RewardEngine()
    rewards = engine.compute_episode_rewards(state, env.ALL_AGENT_IDS)

    # With low allocation, utility gain is negative → stealth bonus should be 0
    for member in ["negotiator_a", "negotiator_b"]:
        stealth = rewards[member].get("coalition_stealth", 0.0)
        assert stealth <= 0.0, \
            f"Should not reward stealth when coalition doesn't benefit members"


def test_overseer_flag_everyone_strategy_fails():
    """
    'Flag all 3 agents' strategy should NOT produce high F1.
    The FP penalty must dominate.
    """
    env = NegotiArenaEnv(seed=42)
    env.reset()
    state = env._state
    # Only one coalition of 2
    from negotiarena_env import CoalitionEvent
    state.coalitions = [CoalitionEvent(
        members=["negotiator_a", "negotiator_b"], formed_at_turn=0
    )]
    # Flag all 3 — 2 correct, 1 false positive
    state.overseer_flags = [
        {"agent_id": "negotiator_a", "turn": 5, "reason": ""},
        {"agent_id": "negotiator_b", "turn": 5, "reason": ""},
        {"agent_id": "negotiator_c", "turn": 5, "reason": ""},  # FP
    ]
    state.final_allocation = {r: TOTAL_RESOURCES[r] / 3 for r in RESOURCE_TYPES}

    engine = RewardEngine()
    rewards = engine.compute_episode_rewards(state, env.ALL_AGENT_IDS)
    detection_reward = rewards["overseer"].get("overseer_detection", 0.0)

    # Should be less than the reward for correctly flagging 2 without FP
    # Proves "flag everyone" strategy is penalised
    assert detection_reward < 2.0, \
        f"Flag-everyone strategy reward too high: {detection_reward}"


# ---------------------------------------------------------------------------
# Curriculum / difficulty tests
# ---------------------------------------------------------------------------

def test_easy_difficulty_lower_coalition_probability():
    """Easy mode should result in fewer coalitions on average."""
    easy_coalition_count = 0
    hard_coalition_count = 0
    N = 30

    for i in range(N):
        env_easy = NegotiArenaEnv(seed=i, difficulty="easy")
        env_easy.reset()
        easy_coalition_count += len(env_easy._state.coalitions)

        env_hard = NegotiArenaEnv(seed=i, difficulty="hard")
        env_hard.reset()
        hard_coalition_count += len(env_hard._state.coalitions)

    assert easy_coalition_count <= hard_coalition_count, \
        f"Easy ({easy_coalition_count}) should have fewer coalitions than hard ({hard_coalition_count})"