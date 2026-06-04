import os
import re
import json
import logging
from typing import Dict, Tuple, List, Any, Optional
from collections import defaultdict
from .base_agent import BaseAgent
from config.prompts import ADVICE_PROMPT, ACTION_SPACE

logger = logging.getLogger(__name__)


class AdviceAgent(BaseAgent):
    """
    Resolves SEEK_ADVICE action into concrete (action, params).
    Not a trainable MAGRPO agent - part of environment system.
    """

    ADVISABLE_ACTIONS = [
        "SHAKE", "ADD_LIGHT", "ADD_NORMAL", "ADD_HEAVY"
    ]

    def __init__(
        self,
        model: str = "llama3.1:8b",
        memory_file: str = "results/advice_memory.json"
    ):
        super().__init__("Advice Agent", model)
        self.memory_file = memory_file
        self.successful_advice: Dict[str, Dict] = {}
        self.advice_history: List[Dict] = []
        self.advice_joint_rewards: Dict[
            str, List[float]
        ] = defaultdict(list)
        self.load_memory()

    def get_advice(
        self,
        current_score: float,
        steps_without_improvement: int,
        ball_distribution: Dict[str, int],
        target_score: float = 0.7
    ) -> Tuple[str, Dict]:
        key = self._build_key(current_score, steps_without_improvement)

        if key in self.successful_advice:
            m = self.successful_advice[key]
            logger.info(f"Memory advice for '{key}': {m['action']}")
            return m['action'], m['params']

        action, params = self._get_llm_advice(
            current_score, steps_without_improvement,
            ball_distribution, target_score
        )

        if action not in self.ADVISABLE_ACTIONS:
            action, params = self._heuristic(
                current_score, steps_without_improvement,
                ball_distribution
            )

        self.advice_history.append({
            "key": key,
            "action": action,
            "params": params,
            "score_before": current_score,
            "score_after": None,
            "outcome_joint_reward": None,
            "steps_stuck": steps_without_improvement
        })

        return action, params

    def _get_llm_advice(
        self,
        score: float,
        steps: int,
        ball_dist: Dict,
        target: float
    ) -> Tuple[str, Dict]:
        prompt = ADVICE_PROMPT.format(
            current_score=score,
            target_score=target,
            steps_without_improvement=steps,
            light_count=ball_dist.get('LIGHT', 0),
            normal_count=ball_dist.get('NORMAL', 0),
            heavy_count=ball_dist.get('HEAVY', 0),
            advice_history=self._format_history()
        )
        response = self.call_llm(prompt)
        if not response:
            return self._heuristic(score, steps, ball_dist)
        return self._parse(response, score, steps, ball_dist)

    def _parse(
        self,
        response: str,
        score: float,
        steps: int,
        ball_dist: Dict
    ) -> Tuple[str, Dict]:
        action = None
        params: Dict = {}

        m = re.search(
            r"RECOMMENDED_ACTION:\s*(\w+)", response, re.I
        )
        if m:
            p = m.group(1).strip().upper()
            if p in self.ADVISABLE_ACTIONS:
                action = p

        pm = re.search(r"PARAMETERS:\s*(.+)", response, re.I)
        if pm and action:
            ps = pm.group(1).strip()
            if action == "SHAKE":
                dm = re.search(r"duration[=\s]+(\d+)", ps, re.I)
                params = {
                    "duration": min(max(int(dm.group(1)), 10), 30)
                } if dm else {"duration": 15}
            elif action in [
                "ADD_LIGHT", "ADD_NORMAL", "ADD_HEAVY"
            ]:
                cm = re.search(r"count[=\s]+(\d+)", ps, re.I)
                params = {
                    "count": min(max(int(cm.group(1)), 1), 5)
                } if cm else {"count": 3}

        if action is None or not params:
            return self._heuristic(score, steps, ball_dist)
        return action, params

    def _heuristic(
        self,
        score: float,
        steps: int,
        ball_dist: Dict
    ) -> Tuple[str, Dict]:
        if score < 0.3:
            return "SHAKE", {"duration": 22}
        elif score < 0.5:
            if steps > 3:
                return "ADD_NORMAL", {"count": 3}
            return "SHAKE", {"duration": 15}
        elif score < 0.7:
            h = ball_dist.get('HEAVY', 0)
            l = ball_dist.get('LIGHT', 0)
            n = ball_dist.get('NORMAL', 0)
            if h > l + n:
                return "ADD_LIGHT", {"count": 2}
            elif l > h + n:
                return "ADD_HEAVY", {"count": 2}
            elif steps > 2:
                return "SHAKE", {"duration": 12}
            return "ADD_NORMAL", {"count": 2}
        return "ADD_LIGHT", {"count": 1}

    def record_outcome(
        self,
        action: str,
        params: Dict,
        score_before: float,
        score_after: float,
        joint_reward: float,
        steps_without_improvement: int = 0
    ):
        key = self._build_key(score_before, steps_without_improvement)
        self.advice_joint_rewards[action].append(joint_reward)

        if self.advice_history:
            self.advice_history[-1]['outcome_joint_reward'] = joint_reward
            self.advice_history[-1]['score_after'] = score_after

        if joint_reward > 0.05:
            existing = self.successful_advice.get(key, {})
            if joint_reward > existing.get('joint_reward', -float('inf')):
                self.successful_advice[key] = {
                    "action": action,
                    "params": params,
                    "joint_reward": joint_reward,
                    "score_before": score_before,
                    "score_after": score_after
                }

        self.save_memory()

    def _build_key(self, score: float, steps: int) -> str:
        return (
            f"score_{round(score*10)/10:.1f}"
            f"_stuck_{min(steps, 5)}"
        )

    def _format_history(self) -> str:
        completed = [
            a for a in self.advice_history[-5:]
            if a.get('outcome_joint_reward') is not None
        ]
        if not completed:
            return "No completed outcomes yet"
        lines = []
        for r in completed:
            reward = r['outcome_joint_reward']
            d = "+" if reward > 0 else ""
            sa = r.get('score_after')
            sa_str = f"{sa:.3f}" if sa is not None else "?"
            lines.append(
                f"{r['action']} | "
                f"{r['score_before']:.3f}→{sa_str} | "
                f"JR:{d}{reward:.3f} "
                f"{'✓' if reward > 0 else '✗'}"
            )
        return "\n".join(lines)

    def save_memory(self):
        try:
            os.makedirs("results", exist_ok=True)
            with open(self.memory_file, 'w') as f:
                json.dump({
                    "successful_advice": self.successful_advice,
                    "advice_history": self.advice_history[-100:],
                    "advice_joint_rewards": {
                        k: v[-50:]
                        for k, v in self.advice_joint_rewards.items()
                    }
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Save failed: {e}")

    def load_memory(self):
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                self.successful_advice = data.get(
                    "successful_advice", {}
                )
                self.advice_history = data.get("advice_history", [])
                for k, v in data.get(
                    "advice_joint_rewards", {}
                ).items():
                    self.advice_joint_rewards[k] = v
        except Exception as e:
            logger.error(f"Load failed: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        total = len(self.advice_history)
        if total == 0:
            return {
                "total_advice_given": 0,
                "memory_size": len(self.successful_advice),
                "reward_type": "joint"
            }
        completed = [
            a for a in self.advice_history
            if a.get('outcome_joint_reward') is not None
        ]
        successful = [
            a for a in completed
            if a.get('outcome_joint_reward', 0) > 0
        ]
        avg_rewards = {
            k: sum(v) / len(v)
            for k, v in self.advice_joint_rewards.items() if v
        }
        return {
            "total_advice_given": total,
            "completed_outcomes": len(completed),
            "successful_advice": len(successful),
            "success_rate": (
                len(successful) / len(completed)
                if completed else 0.0
            ),
            "memory_size": len(self.successful_advice),
            "best_advised_action": (
                max(avg_rewards, key=avg_rewards.get)
                if avg_rewards else "SHAKE"
            ),
            "reward_type": "joint"
        }

    def process(self, input_data: Any) -> Tuple[str, Dict]:
        return self.get_advice(
            current_score=input_data.get('score', 0.0),
            steps_without_improvement=input_data.get(
                'steps_without_improvement', 0
            ),
            ball_distribution=input_data.get('ball_distribution', {}),
            target_score=input_data.get('target_score', 0.7)
        )