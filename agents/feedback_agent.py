"""
agents/feedback_agent.py
"""

import re
import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from .base_agent import BaseAgent
from config.prompts import FEEDBACK_PROMPT

logger = logging.getLogger(__name__)


class FeedbackAgent(BaseAgent):
    """Agent 5: Tracks joint reward outcomes."""

    def __init__(self, model: str = "llama3.1:8b"):
        super().__init__("Feedback Agent", model)
        self.action_history: List[Dict] = []
        self.action_joint_rewards: Dict[
            str, List[float]
        ] = defaultdict(list)
        self.good_actions: set = set()
        self.bad_actions: set = set()

    def process(
        self,
        homogeneity_score: float,
        target_score: float = 0.7,
        best_score: float = 0.0,
        steps_without_improvement: int = 0,
        advice_cooldown: int = 0,
        container_state: list = None
    ) -> Dict[str, Any]:
        light = normal = heavy = 0
        if container_state:
            light = sum(row.count(1) for row in container_state)
            normal = sum(row.count(2) for row in container_state)
            heavy = sum(row.count(3) for row in container_state)

        prompt = FEEDBACK_PROMPT.format(
            action_history=self._format_history(),
            joint_reward_history=self._format_rewards(),
            best_actions=self._format_best(),
            worst_actions=self._format_worst(),
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
            return self._fallback()
        return self._parse(response)

    def record_action(
        self,
        action: str,
        parameters: Dict,
        score_before: float,
        score_after: float,
        joint_reward: float,
        reason: str = ""
    ):
        """Record action with JOINT reward only."""
        record = {
            "action": action,
            "parameters": parameters,
            "score_before": score_before,
            "score_after": score_after,
            "delta": score_after - score_before,
            "joint_reward": joint_reward,
            "reason": reason,
            "step": len(self.action_history)
        }
        self.action_history.append(record)
        self.action_joint_rewards[action].append(joint_reward)

        if joint_reward > 0:
            self.good_actions.add(action)
            self.bad_actions.discard(action)
        elif joint_reward < -0.1:
            self.bad_actions.add(action)
            self.good_actions.discard(action)

    def _format_history(self) -> str:
        if not self.action_history:
            return "No actions yet"
        lines = []
        for r in self.action_history[-10:]:
            d = "+" if r['joint_reward'] > 0 else ""
            lines.append(
                f"Step {r['step']:2}: {r['action']:12} | "
                f"{r['score_before']:.3f}→{r['score_after']:.3f} | "
                f"JR:{d}{r['joint_reward']:.3f}"
            )
        return "\n".join(lines)

    def _format_rewards(self) -> str:
        if not self.action_history:
            return "No reward history"
        recent = [r['joint_reward'] for r in self.action_history[-5:]]
        trend = (
            "↑" if len(recent) > 1 and recent[-1] > recent[0]
            else "↓" if len(recent) > 1 and recent[-1] < recent[0]
            else "→"
        )
        return f"Recent: {[f'{r:.3f}' for r in recent]} {trend}"

    def _format_best(self) -> str:
        if not self.action_joint_rewards:
            return "No data"
        items = sorted(
            self.action_joint_rewards.items(),
            key=lambda x: sum(x[1]) / len(x[1]),
            reverse=True
        )[:3]
        return "\n".join(
            f"{a}: {sum(r)/len(r):+.3f}" for a, r in items
        )

    def _format_worst(self) -> str:
        if not self.action_joint_rewards:
            return "No data"
        items = sorted(
            self.action_joint_rewards.items(),
            key=lambda x: sum(x[1]) / len(x[1])
        )[:2]
        return "\n".join(
            f"{a}: {sum(r)/len(r):+.3f}" for a, r in items
        )

    def _parse(self, response: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "performance_summary": "",
            "pattern_analysis": "",
            "avoid_list": "",
            "recommendation": "",
            "raw_response": response
        }
        sections = {
            "PERFORMANCE SUMMARY": "performance_summary",
            "PATTERN ANALYSIS": "pattern_analysis",
            "AVOID LIST": "avoid_list",
            "RECOMMENDATION": "recommendation"
        }
        for section, key in sections.items():
            pattern = rf"{section}:\s*(.*?)(?=\n[A-Z]|\Z)"
            m = re.search(
                pattern, response, re.DOTALL | re.IGNORECASE
            )
            if m:
                result[key] = m.group(1).strip()[:300]

        defaults = {
            "performance_summary": self.get_action_summary(),
            "pattern_analysis": "Based on joint reward history",
            "avoid_list": str(list(self.bad_actions)),
            "recommendation": self.get_best_action()
        }
        for k, v in defaults.items():
            if not result[k]:
                result[k] = v
        return result

    def _fallback(self) -> Dict[str, Any]:
        return {
            "performance_summary": self.get_action_summary(),
            "pattern_analysis": "Joint reward analysis",
            "avoid_list": str(list(self.bad_actions)),
            "recommendation": self.get_best_action(),
            "raw_response": ""
        }

    def get_action_summary(self) -> str:
        if not self.action_joint_rewards:
            return "No action data yet."
        lines = ["Action Performance (Joint Reward):"]
        for a in ["SHAKE", "ADD_LIGHT", "ADD_NORMAL", "ADD_HEAVY"]:
            rs = self.action_joint_rewards.get(a, [])
            if rs:
                avg = sum(rs) / len(rs)
                sym = "✓" if avg > 0 else "✗"
                lines.append(f"  {sym} {a}: {avg:+.3f} (n={len(rs)})")
            else:
                lines.append(f"  ? {a}: no data")
        return "\n".join(lines)

    def get_best_action(self) -> str:
        if not self.action_joint_rewards:
            return "SHAKE"
        return max(
            self.action_joint_rewards.items(),
            key=lambda x: sum(x[1]) / len(x[1]) if x[1]
            else -float('inf')
        )[0]

    def should_avoid_action(
        self, action: str
    ) -> Tuple[bool, str]:
        if action in self.bad_actions:
            return True, f"Negative joint rewards"
        rs = self.action_joint_rewards.get(action, [])
        if rs and sum(rs) / len(rs) < -0.1:
            return True, f"Low avg joint reward"
        return False, ""

    def get_statistics(self) -> Dict[str, Any]:
        if not self.action_history:
            return {"total_actions": 0, "reward_type": "joint"}
        s = sum(
            1 for a in self.action_history if a['joint_reward'] > 0
        )
        return {
            "total_actions": len(self.action_history),
            "successful_actions": s,
            "success_rate": s / len(self.action_history),
            "best_action": self.get_best_action(),
            "good_actions": list(self.good_actions),
            "bad_actions": list(self.bad_actions),
            "reward_type": "joint"
        }