"""
agents/observation_agent.py
"""

import re
import logging
from typing import Dict, Any
from .base_agent import BaseAgent
from config.prompts import BALL_SYMBOLS, OBSERVATION_PROMPT

logger = logging.getLogger(__name__)


class ObservationAgent(BaseAgent):
    """Agent 1: Observes and analyzes container state."""

    def __init__(self, model: str = "llama3.1:8b"):
        super().__init__("Observation Agent", model)

    def process(
        self,
        container_state: list,
        homogeneity_score: float,
        best_score: float = 0.0,
        target_score: float = 0.7,
        steps_without_improvement: int = 0,
        advice_cooldown: int = 0
    ) -> Dict[str, Any]:
        state_text = self._format_state_visual(container_state)
        light = sum(row.count(1) for row in container_state)
        normal = sum(row.count(2) for row in container_state)
        heavy = sum(row.count(3) for row in container_state)

        prompt = OBSERVATION_PROMPT.format(
            container_state=state_text,
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
            return self._fallback(container_state, homogeneity_score)
        return self._parse(response, homogeneity_score)

    def _format_state_visual(self, state: list) -> str:
        lines = ["Container Grid (10x10):"]
        lines.append("    " + " ".join(f"{i:2}" for i in range(10)))
        lines.append("   " + "-" * 30)
        for i, row in enumerate(state):
            row_text = f"{i:2} | "
            for cell in row:
                row_text += f" {BALL_SYMBOLS.get(cell, '?')}"
            lines.append(row_text)
        return "\n".join(lines)

    def _parse(
        self, response: str, score: float
    ) -> Dict[str, Any]:
        obs = {
            "distribution_summary": "",
            "homogeneity_assessment": "",
            "key_issues": "",
            "notable_patterns": "",
            "recommended_focus": "",
            "raw_response": response,
            "score": score
        }
        keys = {
            "DISTRIBUTION SUMMARY": "distribution_summary",
            "HOMOGENEITY ASSESSMENT": "homogeneity_assessment",
            "KEY ISSUES": "key_issues",
            "NOTABLE PATTERNS": "notable_patterns",
            "RECOMMENDED FOCUS": "recommended_focus"
        }
        current = None
        for line in response.split('\n'):
            matched = False
            for kw, key in keys.items():
                if kw in line.upper():
                    current = key
                    if ':' in line:
                        obs[current] = line.split(':', 1)[-1].strip()
                    matched = True
                    break
            if not matched and current and line.strip():
                obs[current] += ' ' + line.strip()

        defaults = {
            "distribution_summary": "Balls distributed across container",
            "homogeneity_assessment": f"Score: {score:.3f}",
            "key_issues": "Balls not evenly distributed",
            "notable_patterns": "Some clustering observed",
            "recommended_focus": "Focus on most clustered region"
        }
        for k, v in defaults.items():
            if not obs[k]:
                obs[k] = v
        return obs

    def _fallback(
        self, state: list, score: float
    ) -> Dict[str, Any]:
        l = sum(row.count(1) for row in state)
        n = sum(row.count(2) for row in state)
        h = sum(row.count(3) for row in state)
        return {
            "distribution_summary": f"L:{l} N:{n} H:{h}",
            "homogeneity_assessment": f"Score: {score:.3f}",
            "key_issues": "Balls clustered",
            "notable_patterns": "Similar balls grouped",
            "recommended_focus": "Shake to redistribute",
            "raw_response": "",
            "score": score
        }

    def get_observation_summary(self, obs: Dict) -> str:
        return (
            f"Distribution: {obs.get('distribution_summary', 'N/A')}\n"
            f"Assessment: {obs.get('homogeneity_assessment', 'N/A')}\n"
            f"Issues: {obs.get('key_issues', 'N/A')}\n"
            f"Focus: {obs.get('recommended_focus', 'N/A')}"
        )