import re
import logging
from typing import Dict, Any, Tuple
from .base_agent import BaseAgent
from config.prompts import DECISION_PROMPT

logger = logging.getLogger(__name__)


class DecisionAgent(BaseAgent):
    """Agent 3: Makes final action decision."""

    def __init__(self, model: str = "llama3.1:8b"):
        super().__init__("Decision Agent", model)

    def process(
        self,
        reasoning: Dict[str, Any],
        container_state: list,
        homogeneity_score: float,
        target_score: float = 0.7,
        best_score: float = 0.0,
        steps_without_improvement: int = 0,
        advice_cooldown: int = 0
    ) -> Tuple[str, Dict, str]:
        strategy = (
            f"Strategy: {reasoning.get('proposed_strategy', 'N/A')}\n"
            f"Recommended: {reasoning.get('recommended_action', 'N/A')}\n"
            f"Expected: {reasoning.get('expected_outcome', 'N/A')}"
        )

        light = sum(row.count(1) for row in container_state)
        normal = sum(row.count(2) for row in container_state)
        heavy = sum(row.count(3) for row in container_state)

        prompt = DECISION_PROMPT.format(
            strategy_summary=strategy,
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
            return self._fallback(homogeneity_score, advice_cooldown)
        return self._parse(response, advice_cooldown)

    def _parse(
        self, response: str, advice_cooldown: int = 0
    ) -> Tuple[str, Dict, str]:
        valid = [
            "SHAKE", "ADD_LIGHT", "ADD_NORMAL",
            "ADD_HEAVY", "SEEK_ADVICE"
        ]
        action = "SHAKE"
        params: Dict = {"duration": 15}
        reason = ""

        m = re.search(r"ACTION:\s*(\w+)", response, re.IGNORECASE)
        if m:
            parsed = m.group(1).strip().upper()
            if parsed in valid:
                action = parsed

        if action == "SEEK_ADVICE" and advice_cooldown > 0:
            action = "SHAKE"
            params = {"duration": 15}

        p = re.search(r"PARAMETERS:\s*(.+)", response, re.IGNORECASE)
        if p and action != "SEEK_ADVICE":
            ps = p.group(1).strip()
            if action == "SHAKE":
                dm = re.search(r"duration[=\s]+(\d+)", ps, re.I)
                params = {
                    "duration": min(max(int(dm.group(1)), 10), 30)
                } if dm else {"duration": 15}
            elif action in ["ADD_LIGHT", "ADD_NORMAL", "ADD_HEAVY"]:
                cm = re.search(r"count[=\s]+(\d+)", ps, re.I)
                params = {
                    "count": min(max(int(cm.group(1)), 1), 5)
                } if cm else {"count": 3}
        elif action == "SEEK_ADVICE":
            params = {}

        rm = re.search(
            r"REASON:\s*(.+)", response, re.IGNORECASE | re.DOTALL
        )
        if rm:
            reason = rm.group(1).strip()[:200]

        return action, params, reason

    def _fallback(
        self, score: float, advice_cooldown: int = 0
    ) -> Tuple[str, Dict, str]:
        if score < 0.3:
            return "SHAKE", {"duration": 20}, "Low score - vigorous mix"
        elif score < 0.5:
            return "SHAKE", {"duration": 15}, "Medium score - continue"
        elif score < 0.7:
            return "ADD_NORMAL", {"count": 3}, "Good score - add balls"
        else:
            return "ADD_LIGHT", {"count": 2}, "High score - fine tune"