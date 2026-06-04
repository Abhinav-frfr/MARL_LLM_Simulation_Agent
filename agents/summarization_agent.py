"""
agents/summarization_agent.py
"""

import re
import logging
from typing import Dict, Any, List
from .base_agent import BaseAgent
from config.prompts import SUMMARIZATION_PROMPT, BALL_SYMBOLS

logger = logging.getLogger(__name__)


class SummarizationAgent(BaseAgent):
    """Agent 4: Summarizes episode results."""

    def __init__(self, model: str = "llama3.1:8b"):
        super().__init__("Summarization Agent", model)
        self.summary_history: List[Dict] = []

    def process(
        self,
        history: List[Dict],
        final_state: list,
        final_score: float,
        target_score: float = 0.7,
        total_joint_reward: float = 0.0
    ) -> Dict[str, Any]:
        history_text = self._format_history(history)
        state_text = self._format_state(final_state)
        goal = "YES ✓" if final_score >= target_score else "NO ✗"

        prompt = SUMMARIZATION_PROMPT.format(
            simulation_history=history_text,
            final_state=state_text,
            final_score=final_score,
            target_score=target_score,
            goal_achieved=goal,
            total_joint_reward=total_joint_reward,
            total_steps=len(history)
        )

        response = self.call_llm(prompt)
        if not response:
            return self._fallback(
                final_score, len(history), total_joint_reward
            )

        summary = self._parse(
            response, final_score, len(history),
            total_joint_reward, final_score >= target_score
        )
        self.summary_history.append(summary)
        return summary

    def _format_history(self, history: List[Dict]) -> str:
        if not history:
            return "No actions taken."
        lines = []
        for h in history[-15:]:
            jr = h.get('joint_reward', 0.0)
            d = "+" if jr > 0 else ""
            lines.append(
                f"Step {h.get('step','?'):2}: "
                f"{h.get('action','N/A'):12} | "
                f"Score: {h.get('score_after', 0):.3f} | "
                f"JR: {d}{jr:.3f}"
            )
        return "\n".join(lines)

    def _format_state(self, state: list) -> str:
        lines = []
        for row in state[:5]:
            lines.append(" ".join(
                BALL_SYMBOLS.get(c, '?') for c in row[:8]
            ))
        return "\n".join(lines) + "\n...(partial)"

    def _parse(
        self,
        response: str,
        score: float,
        steps: int,
        total_jr: float,
        achieved: bool
    ) -> Dict[str, Any]:
        result = {
            "actions_sequence": "",
            "cooperation_analysis": "",
            "result_analysis": "",
            "recommendations": "",
            "total_steps": steps,
            "final_score": score,
            "total_joint_reward": total_jr,
            "goal_achieved": achieved,
            "raw_response": response
        }
        sections = {
            "ACTIONS SEQUENCE": "actions_sequence",
            "COOPERATION ANALYSIS": "cooperation_analysis",
            "RESULT ANALYSIS": "result_analysis",
            "RECOMMENDATIONS": "recommendations"
        }
        for section, key in sections.items():
            pattern = rf"{section}:\s*(.*?)(?=\n[A-Z]|\Z)"
            m = re.search(
                pattern, response, re.DOTALL | re.IGNORECASE
            )
            if m:
                result[key] = m.group(1).strip()[:300]

        defaults = {
            "actions_sequence": f"Took {steps} steps",
            "cooperation_analysis": f"Joint reward: {total_jr:.3f}",
            "result_analysis": (
                f"Score {score:.3f} "
                f"({'met' if achieved else 'not met'})"
            ),
            "recommendations": "Adjust strategy"
        }
        for k, v in defaults.items():
            if not result[k]:
                result[k] = v
        return result

    def _fallback(
        self, score: float, steps: int, total_jr: float
    ) -> Dict[str, Any]:
        return {
            "actions_sequence": f"Took {steps} steps",
            "cooperation_analysis": f"Joint reward: {total_jr:.3f}",
            "result_analysis": f"Score {score:.3f}",
            "recommendations": "Adjust strategy",
            "total_steps": steps,
            "final_score": score,
            "total_joint_reward": total_jr,
            "goal_achieved": score >= 0.7,
            "raw_response": ""
        }

    def generate_episode_summary(
        self,
        episode_num: int,
        best_score: float,
        actions_taken: List[str],
        total_joint_reward: float,
        duration: float
    ) -> str:
        from collections import Counter
        counts = Counter(actions_taken)
        most_common = (
            counts.most_common(1)[0][0] if counts else "SHAKE"
        )
        return (
            f"Episode {episode_num}:\n"
            f"  Best Score: {best_score:.3f}\n"
            f"  Most Used: {most_common}\n"
            f"  Actions: {len(actions_taken)}\n"
            f"  Joint Reward: {total_joint_reward:.3f}\n"
            f"  Duration: {duration:.1f}s"
        )