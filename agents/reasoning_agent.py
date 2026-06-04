"""
agents/reasoning_agent.py
"""

import re
import logging
from typing import Dict, Any, Optional
from .base_agent import BaseAgent
from config.prompts import REASONING_PROMPT

logger = logging.getLogger(__name__)


class ReasoningAgent(BaseAgent):
    """Agent 2: Reasons about situation and proposes strategy."""

    def __init__(self, model: str = "llama3.1:8b"):
        super().__init__("Reasoning Agent", model)
        self.feedback_agent = None

    def set_feedback_agent(self, feedback_agent):
        self.feedback_agent = feedback_agent

    def process(
        self,
        observation: Dict[str, Any],
        container_state: list,
        homogeneity_score: float,
        target_score: float = 0.7,
        best_score: float = 0.0,
        steps_without_improvement: int = 0,
        advice_cooldown: int = 0,
        last_action: str = "None",
        last_joint_reward: float = 0.0
    ) -> Dict[str, Any]:
        obs_summary = (
            f"Distribution: "
            f"{observation.get('distribution_summary', 'N/A')}\n"
            f"Assessment: "
            f"{observation.get('homogeneity_assessment', 'N/A')}\n"
            f"Issues: {observation.get('key_issues', 'N/A')}\n"
            f"Focus: {observation.get('recommended_focus', 'N/A')}"
        )

        if last_action == "None":
            last_outcome = "No previous action (first step)"
        else:
            d = "+" if last_joint_reward > 0 else ""
            q = ("GOOD ✓" if last_joint_reward > 0
                 else "BAD ✗" if last_joint_reward < 0
                 else "NEUTRAL")
            last_outcome = (
                f"Action: {last_action} | "
                f"Joint Reward: {d}{last_joint_reward:.3f} | {q}"
            )

        feedback_summary = (
            self.feedback_agent.get_action_summary()
            if self.feedback_agent and
            self.feedback_agent.action_history
            else "No feedback yet"
        )

        light = sum(row.count(1) for row in container_state)
        normal = sum(row.count(2) for row in container_state)
        heavy = sum(row.count(3) for row in container_state)

        prompt = REASONING_PROMPT.format(
            observation_summary=obs_summary,
            last_action_outcome=last_outcome,
            feedback_summary=feedback_summary,
            homogeneity_score=homogeneity_score,
            target_score=target_score,
            best_score=best_score,
            steps_without_improvement=steps_without_improvement,
            advice_cooldown=advice_cooldown,
            light_count=light,
            normal_count=normal,
            heavy_count=heavy
        )

        response = self.call_llm(prompt)
        if not response:
            return self._fallback(homogeneity_score)
        return self._parse(response)

    def _parse(self, response: str) -> Dict[str, Any]:
        result = {
            "situation_analysis": "",
            "proposed_strategy": "",
            "recommended_action": "",
            "expected_outcome": "",
            "cooperation_note": "",
            "raw_response": response
        }
        sections = {
            "SITUATION ANALYSIS": "situation_analysis",
            "STRATEGY": "proposed_strategy",
            "RECOMMENDED ACTION": "recommended_action",
            "EXPECTED OUTCOME": "expected_outcome",
            "COOPERATION NOTE": "cooperation_note"
        }
        for section, key in sections.items():
            pattern = rf"{section}:\s*(.*?)(?=\n[A-Z]|\Z)"
            m = re.search(
                pattern, response, re.DOTALL | re.IGNORECASE
            )
            if m:
                result[key] = m.group(1).strip()[:300]

        defaults = {
            "situation_analysis": "Need better distribution",
            "proposed_strategy": "Shake to mix",
            "recommended_action": "SHAKE",
            "expected_outcome": "Score improvement",
            "cooperation_note": "Joint reward improves with mixing"
        }
        for k, v in defaults.items():
            if not result[k]:
                result[k] = v
        return result

    def _fallback(self, score: float) -> Dict[str, Any]:
        action = "SHAKE" if score < 0.5 else "ADD_NORMAL"
        return {
            "situation_analysis": f"Score {score:.3f}",
            "proposed_strategy": "Continue mixing",
            "recommended_action": action,
            "expected_outcome": "Score improvement",
            "cooperation_note": "Shared reward",
            "raw_response": ""
        }

    def get_reasoning_summary(self, r: Dict) -> str:
        return (
            f"Strategy: {r.get('proposed_strategy', 'N/A')}\n"
            f"Recommended: {r.get('recommended_action', 'N/A')}\n"
            f"Expected: {r.get('expected_outcome', 'N/A')}"
        )