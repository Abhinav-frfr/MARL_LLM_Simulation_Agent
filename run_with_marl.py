"""
run_with_marl.py - Inference Entry Point
"""

import os
import json
import copy
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional
from colorama import init
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from simulation.container_simulation import ContainerSimulation
from agents.observation_agent import ObservationAgent
from agents.reasoning_agent import ReasoningAgent
from agents.decision_agent import DecisionAgent
from agents.summarization_agent import SummarizationAgent
from agents.feedback_agent import FeedbackAgent
from agents.advice_agent import AdviceAgent
from marl.magrpo_trainer import MAGRPOTrainer
from config.prompts import BALL_SYMBOLS

init(autoreset=True)
console = Console()


class MARLInferenceController:

    AGENT_NAMES = [
        'observation', 'reasoning', 'decision',
        'summarization', 'feedback'
    ]

    def __init__(
        self,
        model_path: str = "results/magrpo_final_model.pt",
        max_steps: int = 20,
        target_score: float = 0.7,
        model: str = "llama3.1:8b",
        G: int = 8
    ):
        self.simulation = ContainerSimulation()
        self.max_steps = max_steps
        self.target_score = target_score
        self.G = G

        self.llm_agents = {
            'observation':   ObservationAgent(model),
            'reasoning':     ReasoningAgent(model),
            'decision':      DecisionAgent(model),
            'summarization': SummarizationAgent(model),
            'feedback':      FeedbackAgent(model)
        }
        self.llm_agents['reasoning'].set_feedback_agent(
            self.llm_agents['feedback']
        )
        self.advice_agent = AdviceAgent(model)

        self.trainer = MAGRPOTrainer(
            state_dim=64,
            num_agents=len(self.AGENT_NAMES),
            action_dim=5,
            lr=3e-4,
            G=G
        )

        if os.path.exists(model_path):
            self.trainer.load_model(model_path)
            console.print(f"[green]✓ Loaded: {model_path}[/green]")
        else:
            console.print(
                f"[yellow]⚠ No model at {model_path}[/yellow]"
            )

        self.history: List[Dict] = []
        self.best_score = 0.0
        self.steps_without_improvement = 0
        self.advice_cooldown = 0
        self.joint_rewards_log: List[float] = []
        self.last_advice_action: Optional[str] = None
        self.last_advice_params: Optional[Dict] = None
        self.last_advice_score: Optional[float] = None

    def get_agent_state(
        self, score: float, step: int
    ) -> Dict[str, Any]:
        return {
            'score': score,
            'ball_counts': self.simulation.get_ball_distribution(),
            'best_score': self.best_score,
            'steps_without_improvement': self.steps_without_improvement,
            'advice_cooldown': self.advice_cooldown,
            'step': step
        }

    def compute_joint_reward(
        self, before: float, after: float
    ) -> float:
        return self.simulation.get_joint_reward(
            score_before=before,
            score_after=after,
            target_score=self.target_score,
            steps_without_improvement=self.steps_without_improvement
        )

    def resolve_action(
        self, action_idx: int, score: float
    ) -> Tuple[str, Dict, bool]:
        raw = MAGRPOTrainer.ACTION_SPACE[action_idx]
        if raw == "SHAKE":
            p = (
                {"duration": 20} if score < 0.3
                else {"duration": 15} if score < 0.5
                else {"duration": 10}
            )
            return raw, p, False
        elif raw in ["ADD_LIGHT", "ADD_NORMAL", "ADD_HEAVY"]:
            p = (
                {"count": 4} if score < 0.4
                else {"count": 3} if score < 0.6
                else {"count": 2}
            )
            return raw, p, False
        elif raw == "SEEK_ADVICE":
            if self.advice_cooldown > 0:
                return "SHAKE", {"duration": 15}, False
            bd = self.simulation.get_ball_distribution()
            a, p = self.advice_agent.get_advice(
                current_score=score,
                steps_without_improvement=self.steps_without_improvement,
                ball_distribution=bd,
                target_score=self.target_score
            )
            self.last_advice_action = a
            self.last_advice_params = p
            self.last_advice_score = score
            self.advice_cooldown = 3
            return a, p, True
        return "SHAKE", {"duration": 15}, False

    def select_best_group(
        self,
        group_actions: List[List[int]],
        score: float
    ) -> Tuple[str, Dict, int, bool]:
        backup = self.simulation.copy_state()
        best_delta = -float('inf')
        best_name, best_params = "SHAKE", {"duration": 15}
        best_g, best_is_adv = 0, False
        adv_resolved = False

        for g in range(self.G):
            self.simulation.restore_state(copy.deepcopy(backup))
            name, params, is_adv = self.resolve_action(
                group_actions[g][2], score
            )
            if is_adv and adv_resolved:
                is_adv = False
            if is_adv:
                adv_resolved = True

            self.simulation.execute_action(name, params)
            trial = self.simulation.calculate_homogeneity()
            delta = trial - score

            if delta > best_delta:
                best_delta = delta
                best_name = name
                best_params = params
                best_g = g
                best_is_adv = is_adv

        self.simulation.restore_state(copy.deepcopy(backup))
        return best_name, best_params, best_g, best_is_adv

    def run(self) -> float:
        self.simulation.reset()
        self.simulation.add_multiple_balls(1, 8)
        self.simulation.add_multiple_balls(2, 8)
        self.simulation.add_multiple_balls(3, 8)

        self.history = []
        self.best_score = 0.0
        self.steps_without_improvement = 0
        self.advice_cooldown = 0
        self.joint_rewards_log = []
        self.last_advice_action = None
        self.last_advice_params = None
        self.last_advice_score = None

        for step in range(self.max_steps):
            score = self.simulation.calculate_homogeneity()

            if score > self.best_score:
                self.best_score = score
                self.steps_without_improvement = 0
            else:
                self.steps_without_improvement += 1

            self._display_state(step, score)

            if score >= self.target_score:
                console.print(
                    f"\n[bold green]✓ Goal step {step+1}! "
                    f"{score:.3f}[/bold green]"
                )
                break

            states = [
                self.get_agent_state(score, step)
                for _ in self.AGENT_NAMES
            ]

            group_actions, _ = \
                self.trainer.generate_group_actions(
                    states, temperature=0.1
                )

            name, params, best_g, is_adv = \
                self.select_best_group(group_actions, score)

            self._display_decisions(group_actions[best_g])
            console.print(f"\n[yellow]→ {name} {params}[/yellow]")

            self.simulation.execute_action(name, params)
            new_score = self.simulation.calculate_homogeneity()

            jr = self.compute_joint_reward(score, new_score)
            self.joint_rewards_log.append(jr)
            delta = new_score - score

            self.llm_agents['feedback'].record_action(
                action=name, parameters=params,
                score_before=score, score_after=new_score,
                joint_reward=jr
            )

            if is_adv and self.last_advice_action is not None:
                self.advice_agent.record_outcome(
                    action=self.last_advice_action,
                    params=self.last_advice_params or {},
                    score_before=self.last_advice_score or score,
                    score_after=new_score,
                    joint_reward=jr,
                    steps_without_improvement=self.steps_without_improvement
                )
                self.last_advice_action = None
                self.last_advice_params = None
                self.last_advice_score = None

            color = (
                "green" if delta > 0
                else "red" if delta < 0
                else "yellow"
            )
            console.print(
                f"   [{color}]Δ{delta:+.3f}[/{color}]"
            )

            self.history.append({
                "step": step + 1,
                "action": name,
                "score_before": score,
                "score_after": new_score,
                "delta": delta,
                "joint_reward": jr,
                "advice_used": is_adv
            })

            if self.advice_cooldown > 0:
                self.advice_cooldown -= 1

            if self.steps_without_improvement >= 5:
                console.print("[yellow]⚠ Stagnation[/yellow]")
                self.simulation.add_multiple_balls(2, 3)

        self._display_summary()
        self._save_log()
        return self.best_score

    def _display_decisions(self, actions: List[int]):
        icons = ['👁', '🧠', '🎯', '📝', '📊']
        console.print("\n[bold cyan]Decisions:[/bold cyan]")
        for i, (n, idx) in enumerate(
            zip(self.AGENT_NAMES, actions)
        ):
            console.print(
                f"  {icons[i]} {n:15} → "
                f"{MAGRPOTrainer.ACTION_SPACE[idx]}"
            )

    def _display_state(self, step: int, score: float):
        console.print(f"\n[dim]{'─'*50}[/dim]")
        console.print(
            f"[bold]Step {step+1}/{self.max_steps}[/bold] | "
            f"Best:{self.best_score:.3f} | "
            f"Stuck:{self.steps_without_improvement}"
        )
        table = Table(show_header=False)
        table.add_column("", style="dim")
        table.add_column("")
        for i, row in enumerate(self.simulation.get_state()[:6]):
            row_str = " ".join(
                BALL_SYMBOLS.get(c, '?') for c in row[:8]
            )
            table.add_row(f"R{i}", row_str)
        console.print(table)
        color = (
            "green" if score >= self.target_score
            else "yellow" if score >= 0.5 else "red"
        )
        console.print(f"[{color}]Score:{score:.3f}[/{color}]")
        d = self.simulation.get_ball_distribution()
        console.print(
            f"L={d['LIGHT']} N={d['NORMAL']} H={d['HEAVY']}"
        )

    def _display_summary(self):
        if not self.history:
            return
        table = Table(title="Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row(
            "Initial", f"{self.history[0]['score_before']:.3f}"
        )
        table.add_row(
            "Final", f"{self.history[-1]['score_after']:.3f}"
        )
        table.add_row("Best", f"{self.best_score:.3f}")
        table.add_row("Steps", str(len(self.history)))
        table.add_row(
            "Joint Reward",
            f"{sum(self.joint_rewards_log):.3f}"
        )
        table.add_row(
            "Advice Used",
            str(sum(1 for h in self.history if h['advice_used']))
        )
        table.add_row(
            "Goal",
            "✓ YES" if self.best_score >= self.target_score
            else "✗ NO"
        )
        console.print(table)
        console.print(
            f"\n{self.llm_agents['feedback'].get_action_summary()}"
        )

    def _save_log(self):
        os.makedirs("results", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"results/inference_{ts}.json"
        with open(fn, 'w') as f:
            json.dump({
                "timestamp": ts,
                "config": {
                    "G": self.G, "H": self.max_steps,
                    "target": self.target_score
                },
                "results": {
                    "best_score": self.best_score,
                    "goal": self.best_score >= self.target_score,
                    "steps": len(self.history),
                    "joint_reward": sum(self.joint_rewards_log)
                },
                "steps": [
                    {
                        "step": h["step"],
                        "action": h["action"],
                        "score_before": h["score_before"],
                        "score_after": h["score_after"],
                        "joint_reward": h["joint_reward"],
                        "advice_used": h["advice_used"]
                    }
                    for h in self.history
                ]
            }, f, indent=2)
        console.print(f"[green]✓ {fn}[/green]")


def main():
    path = "results/magrpo_final_model.pt"
    if not os.path.exists(path):
        console.print(f"[yellow]No model. Run: python run_marl.py[/yellow]")
        return

    ctrl = MARLInferenceController(
        model_path=path, max_steps=20,
        target_score=0.7, model="llama3.1:8b", G=8
    )
    try:
        best = ctrl.run()
        if best >= 0.7:
            console.print("[bold green]✓ SUCCESS![/bold green]")
        else:
            console.print(f"[yellow]Best:{best:.3f}[/yellow]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]{e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()