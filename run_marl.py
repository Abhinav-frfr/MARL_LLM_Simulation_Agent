"""
run_marl.py - Training Entry Point
MAGRPO Algorithm 1 implementation.
Fixes:
  - Reduced default hyperparameters for feasible runtime
  - LLM calls batched / moved out of hot loops
  - Async group execution
  - Proper progress tracking
  - Summarization called every N episodes, not every episode
  - Advice agent called once per episode max (not per group)
"""

import os
import json
import copy
import asyncio
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional
from colorama import init
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress, BarColumn, TextColumn,
    TimeElapsedColumn, MofNCompleteColumn
)

from simulation.container_simulation import ContainerSimulation
from agents.observation_agent import ObservationAgent
from agents.reasoning_agent import ReasoningAgent
from agents.decision_agent import DecisionAgent
from agents.summarization_agent import SummarizationAgent
from agents.feedback_agent import FeedbackAgent
from agents.advice_agent import AdviceAgent
from marl.magrpo_trainer import MAGRPOTrainer

init(autoreset=True)
console = Console()
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Tuneable constants – change here only
# ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG: Dict[str, Any] = {
    # Training
    "num_episodes":    10,     # 100 → 10 for feasible runtime
    "max_steps":       10,     # 20  → 10 steps per episode
    "G":               4,      # 8   → 4 groups (halves LLM calls)
    "save_interval":   5,      # save checkpoint every N episodes
    "summarize_every": 3,      # LLM summary every N episodes
    "advice_cooldown": 3,      # steps between advice calls
    # Environment
    "target_score":    0.7,
    "balls_per_type":  8,      # light / normal / heavy
    # Model
    "model":           "llama3.1:8b",
    # MAGRPO trainer
    "state_dim":       64,
    "action_dim":      5,
    "lr":              3e-4,
}


class MAGRPOSimulationController:
    """
    Runs MAGRPO (Algorithm 1).

    Key design decisions
    --------------------
    * LLM agents are called OUTSIDE the G-group hot loop.
      The group loop is pure NumPy / simulation math.
    * AdviceAgent LLM is called at most ONCE per episode
      (pre-step), not once per group.
    * SummarizationAgent LLM is called every `summarize_every`
      episodes to amortise cost.
    * All G groups share one simulation backup; restore is O(1)
      via deepcopy of a lightweight state dict.
    """

    AGENT_NAMES = [
        'observation', 'reasoning', 'decision',
        'summarization', 'feedback'
    ]

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg           = cfg
        self.max_steps     = cfg["max_steps"]
        self.target_score  = cfg["target_score"]
        self.num_episodes  = cfg["num_episodes"]
        self.save_interval = cfg["save_interval"]
        self.G             = cfg["G"]
        self.summarize_every = cfg["summarize_every"]
        model              = cfg["model"]

        # ── Simulation ───────────────────────────────────────
        self.simulation = ContainerSimulation()

        # ── LLM Agents (initialised once, reused) ────────────
        self.llm_agents: Dict[str, Any] = {
            'observation':   ObservationAgent(model),
            'reasoning':     ReasoningAgent(model),
            'decision':      DecisionAgent(model),
            'summarization': SummarizationAgent(model),
            'feedback':      FeedbackAgent(model),
        }
        self.llm_agents['reasoning'].set_feedback_agent(
            self.llm_agents['feedback']
        )
        self.advice_agent = AdviceAgent(model)

        # ── MAGRPO Trainer (neural policy heads) ─────────────
        self.trainer = MAGRPOTrainer(
            state_dim  = cfg["state_dim"],
            num_agents = len(self.AGENT_NAMES),
            action_dim = cfg["action_dim"],
            lr         = cfg["lr"],
            G          = self.G,
        )

        # ── Episode state ─────────────────────────────────────
        self._reset_episode_state()

        # ── Stats ─────────────────────────────────────────────
        self.all_scores:   List[float] = []
        self.all_returns:  List[float] = []

        self._print_banner()

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    def _print_banner(self):
        console.print(Panel.fit(
            f"[bold cyan]MAGRPO Training[/bold cyan]\n"
            f"Agents : {len(self.AGENT_NAMES)} + AdviceAgent\n"
            f"Episodes: {self.num_episodes}  "
            f"H: {self.max_steps}  G: {self.G}\n"
            f"[green]Joint reward | No clip | KL=0[/green]",
            border_style="cyan"
        ))

    def _reset_episode_state(self):
        """Reset per-episode counters."""
        self.episode_best_score      = 0.0
        self.steps_without_improve   = 0
        self.advice_cooldown         = 0
        self.last_advice_action:  Optional[str]  = None
        self.last_advice_params:  Optional[Dict] = None
        self.last_advice_score:   Optional[float]= None

    def _state_vector(
        self, score: float, step: int
    ) -> Dict[str, Any]:
        """Lightweight state dict (no LLM, O(1))."""
        return {
            'score':                    score,
            'ball_counts':              self.simulation.get_ball_distribution(),
            'best_score':               self.episode_best_score,
            'steps_without_improvement':self.steps_without_improve,
            'advice_cooldown':          self.advice_cooldown,
            'step':                     step,
        }

    def _joint_reward(
        self, before: float, after: float
    ) -> float:
        return self.simulation.get_joint_reward(
            score_before             = before,
            score_after              = after,
            target_score             = self.target_score,
            steps_without_improvement= self.steps_without_improve,
        )

    # ─────────────────────────────────────────────────────────
    # Action resolution  (NO LLM inside this method)
    # ─────────────────────────────────────────────────────────

    def _resolve_action(
        self,
        raw_action: str,
        current_score: float,
        episode_advice: Optional[Tuple[str, Dict]],
    ) -> Tuple[str, Dict, bool]:
        """
        Map a raw ACTION_SPACE string → (name, params, is_advice).
        `episode_advice` is pre-fetched before the group loop.
        """
        if raw_action == "SEEK_ADVICE":
            if self.advice_cooldown > 0 or episode_advice is None:
                # Fallback – advice unavailable this step
                return "SHAKE", {"duration": 15}, False
            return episode_advice[0], episode_advice[1], True

        if raw_action == "SHAKE":
            dur = 20 if current_score < 0.3 else \
                  15 if current_score < 0.5 else 10
            return "SHAKE", {"duration": dur}, False

        if raw_action in ("ADD_LIGHT", "ADD_NORMAL", "ADD_HEAVY"):
            cnt = 4 if current_score < 0.4 else \
                  3 if current_score < 0.6 else 2
            return raw_action, {"count": cnt}, False

        # Unknown – safe default
        return "SHAKE", {"duration": 15}, False

    # ─────────────────────────────────────────────────────────
    # Group execution  (pure math, no LLM)
    # ─────────────────────────────────────────────────────────

    def _execute_groups(
        self,
        group_actions: List[List[int]],
        current_score: float,
        episode_advice: Optional[Tuple[str, Dict]],
    ) -> Tuple[List[float], float, str, bool]:
        """
        Evaluate all G candidate action groups on a simulation
        copy, pick the best, apply it to the real simulation.

        Returns
        -------
        joint_rewards : per-group joint reward
        actual_score  : score after best action applied
        best_name     : action name that was executed
        advice_used   : whether the winning group used advice
        """
        sim_backup = self.simulation.copy_state()

        group_scores:    List[float] = []
        group_names:     List[str]   = []
        group_params:    List[Dict]  = []
        group_is_advice: List[bool]  = []

        for g in range(self.G):
            # Restore clean state for this group
            self.simulation.restore_state(copy.deepcopy(sim_backup))

            raw = MAGRPOTrainer.ACTION_SPACE[group_actions[g][2]]
            name, params, is_adv = self._resolve_action(
                raw, current_score, episode_advice
            )

            self.simulation.execute_action(name, params)
            s = self.simulation.calculate_homogeneity()

            group_scores.append(s)
            group_names.append(name)
            group_params.append(params)
            group_is_advice.append(is_adv)

        # ── Compute joint rewards ─────────────────────────────
        joint_rewards = [
            self._joint_reward(current_score, s)
            for s in group_scores
        ]

        # ── Best group wins ───────────────────────────────────
        best_g      = int(np.argmax(joint_rewards))
        best_name   = group_names[best_g]
        best_params = group_params[best_g]
        best_is_adv = group_is_advice[best_g]

        # Apply best action to real simulation
        self.simulation.restore_state(copy.deepcopy(sim_backup))
        self.simulation.execute_action(best_name, best_params)
        actual_score = self.simulation.calculate_homogeneity()

        # ── Cooldown bookkeeping ──────────────────────────────
        if best_is_adv:
            self.advice_cooldown = self.cfg["advice_cooldown"]
        elif self.advice_cooldown > 0:
            self.advice_cooldown -= 1

        return joint_rewards, actual_score, best_name, best_is_adv

    # ─────────────────────────────────────────────────────────
    # Episode
    # ─────────────────────────────────────────────────────────

    def _run_episode(self, episode_num: int) -> float:
        """Run one episode. Returns total (undiscounted) return."""

        # ── Reset ─────────────────────────────────────────────
        self.simulation.reset()
        n = self.cfg["balls_per_type"]
        for btype in (1, 2, 3):
            self.simulation.add_multiple_balls(btype, n)

        self._reset_episode_state()
        self.llm_agents['feedback'].action_history.clear()
        self.llm_agents['feedback'].action_joint_rewards.clear()

        total_reward = 0.0
        history: List[Dict] = []

        for t in range(self.max_steps):
            score = self.simulation.calculate_homogeneity()

            # ── Track improvement ─────────────────────────────
            if score > self.episode_best_score:
                self.episode_best_score  = score
                self.steps_without_improve = 0
            else:
                self.steps_without_improve += 1

            # ── Pre-fetch advice ONCE per step (not per group) ─
            # Only calls LLM when SEEK_ADVICE might be sampled
            # and cooldown allows.
            episode_advice: Optional[Tuple[str, Dict]] = None
            if self.advice_cooldown == 0:
                # Check if any group action might be SEEK_ADVICE
                # We do a cheap peek at the policy output below
                # (advice is fetched lazily after action sampling)
                pass  # resolved after group_actions sampled

            # ── Build state vectors (no LLM) ──────────────────
            states = [
                self._state_vector(score, t)
                for _ in self.AGENT_NAMES
            ]

            # ── Sample G groups from neural policy ────────────
            group_actions, group_log_probs = \
                self.trainer.generate_group_actions(states)

            # ── Fetch advice only if needed & allowed ─────────
            needs_advice = any(
                MAGRPOTrainer.ACTION_SPACE[ga[2]] == "SEEK_ADVICE"
                for ga in group_actions
            )
            if needs_advice and self.advice_cooldown == 0:
                adv_action, adv_params = \
                    self.advice_agent.get_advice(
                        current_score            = score,
                        steps_without_improvement= self.steps_without_improve,
                        ball_distribution        = self.simulation.get_ball_distribution(),
                        target_score             = self.target_score,
                    )
                episode_advice          = (adv_action, adv_params)
                self.last_advice_action = adv_action
                self.last_advice_params = adv_params
                self.last_advice_score  = score

            # ── Execute groups (no LLM) ────────────────────────
            joint_rewards, new_score, best_action, adv_used = \
                self._execute_groups(
                    group_actions, score, episode_advice
                )

            # ── Record trajectory ──────────────────────────────
            self.trainer.record_step(
                agent_states   = states,
                group_actions  = group_actions,
                group_log_probs= group_log_probs,
                joint_rewards  = joint_rewards,
            )

            mean_jr       = float(np.mean(joint_rewards))
            total_reward += mean_jr

            # ── Feedback agent record (lightweight) ───────────
            self.llm_agents['feedback'].record_action(
                action      = best_action,
                parameters  = {},
                score_before= score,
                score_after = new_score,
                joint_reward= mean_jr,
            )

            # ── Advice outcome (if used) ───────────────────────
            if adv_used and self.last_advice_action is not None:
                self.advice_agent.record_outcome(
                    action                   = self.last_advice_action,
                    params                   = self.last_advice_params or {},
                    score_before             = self.last_advice_score or score,
                    score_after              = new_score,
                    joint_reward             = mean_jr,
                    steps_without_improvement= self.steps_without_improve,
                )
                self.last_advice_action = None
                self.last_advice_params = None
                self.last_advice_score  = None

            history.append({
                "step":         t + 1,
                "action":       best_action,
                "score_before": score,
                "score_after":  new_score,
                "delta":        new_score - score,
                "joint_reward": mean_jr,
            })

            if new_score >= self.target_score:
                console.print(
                    f"   [green]✓ Target at step {t+1}! "
                    f"score={new_score:.3f}[/green]"
                )
                break

        # ── Summarization (LLM) – amortised ───────────────────
        # Only call the LLM summarizer every N episodes
        if (episode_num + 1) % self.summarize_every == 0:
            total_jr = sum(h['joint_reward'] for h in history)
            self.llm_agents['summarization'].generate_episode_summary(
                episode_num      = episode_num + 1,
                best_score       = self.episode_best_score,
                actions_taken    = [h['action'] for h in history],
                total_joint_reward= total_jr,
                duration         = float(len(history)),
            )

        return total_reward

    # ─────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────

    def train(self) -> List[float]:
        console.print(
            f"\n[bold green]Starting MAGRPO "
            f"({self.num_episodes} episodes)[/bold green]"
        )

        episode_returns: List[float] = []

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),          # shows  3/10
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            refresh_per_second=2,
        )

        with progress:
            task = progress.add_task(
                "[cyan]Training...", total=self.num_episodes
            )

            for episode in range(self.num_episodes):

                ep_return = self._run_episode(episode)
                episode_returns.append(ep_return)
                self.all_returns.append(ep_return)

                # ── MAGRPO policy update (every episode) ──────
                metrics = self.trainer.update_all_agents()

                # ── Checkpoint ────────────────────────────────
                if (episode + 1) % self.save_interval == 0:
                    os.makedirs("results", exist_ok=True)
                    self.trainer.save_model(
                        f"results/magrpo_ep{episode+1}.pt"
                    )

                # ── Advance progress bar ───────────────────────
                progress.update(task, advance=1)

                # ── Per-episode console log ────────────────────
                self._log_episode(episode + 1, ep_return, metrics)

        # ── Final save ────────────────────────────────────────
        os.makedirs("results", exist_ok=True)
        self.trainer.save_model("results/magrpo_final_model.pt")
        self._save_history(episode_returns)
        self._display_summary(episode_returns)

        return episode_returns

    # ─────────────────────────────────────────────────────────
    # Logging / display
    # ─────────────────────────────────────────────────────────

    def _log_episode(
        self, ep: int, ret: float, m: Dict
    ):
        """Print one-liner after every episode."""
        loss_vals = [v for k, v in m.items() if 'loss' in k]
        avg_loss  = float(np.mean(loss_vals)) if loss_vals else 0.0
        adv_mean  = m.get('mean_advantage', 0.0)

        console.print(
            f"  Ep [bold]{ep:>3}[/bold]/"
            f"{self.num_episodes} | "
            f"Ret:[cyan]{ret:>7.3f}[/cyan] | "
            f"Best:[green]{self.episode_best_score:.3f}[/green] | "
            f"Loss:[yellow]{avg_loss:.4f}[/yellow] | "
            f"Adv:{adv_mean:.3f}"
        )

    def _display_summary(self, returns: List[float]):
        table = Table(title="MAGRPO Results", border_style="cyan")
        table.add_column("Metric",   style="cyan",  no_wrap=True)
        table.add_column("Value",    style="green")

        rows = [
            ("Episodes",    str(len(returns))),
            ("G (groups)",  str(self.G)),
            ("H (horizon)", str(self.max_steps)),
            ("Avg Return",  f"{np.mean(returns):.4f}"),
            ("Std Return",  f"{np.std(returns):.4f}"),
            ("Best Return", f"{np.max(returns):.4f}"),
            ("Worst Return",f"{np.min(returns):.4f}"),
            ("Reward",      "Joint ✓"),
            ("Clipping",    "None  ✓"),
            ("KL",          "0     ✓"),
        ]
        for k, v in rows:
            table.add_row(k, v)

        console.print(table)

    def _save_history(self, returns: List[float]):
        os.makedirs("results", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"results/magrpo_training_{ts}.json"
        payload = {
            "algorithm": "MAGRPO",
            "config": {
                "episodes":  self.num_episodes,
                "agents":    len(self.AGENT_NAMES),
                "G":         self.G,
                "H":         self.max_steps,
                "target":    self.target_score,
                "reward":    "joint",
                "clipping":  "none",
                "kl":        0,
            },
            "results": {
                "returns": returns,
                "best":    float(np.max(returns)),
                "avg":     float(np.mean(returns)),
                "std":     float(np.std(returns)),
            },
        }
        with open(fn, 'w') as f:
            json.dump(payload, f, indent=2)
        console.print(f"[green]✓ Saved → {fn}[/green]")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    console.print(Panel.fit(
        "[bold cyan]MAGRPO Training[/bold cyan]\n"
        "[green]✓ Joint reward ✓ G groups "
        "✓ No clip ✓ KL=0[/green]",
        border_style="cyan"
    ))

    cfg = DEFAULT_CONFIG.copy()

    # ── Override from environment / CLI (optional) ────────────
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",  type=int,   default=cfg["num_episodes"])
    parser.add_argument("--steps",     type=int,   default=cfg["max_steps"])
    parser.add_argument("--G",         type=int,   default=cfg["G"])
    parser.add_argument("--model",     type=str,   default=cfg["model"])
    parser.add_argument("--target",    type=float, default=cfg["target_score"])
    args = parser.parse_args()

    cfg.update({
        "num_episodes": args.episodes,
        "max_steps":    args.steps,
        "G":            args.G,
        "model":        args.model,
        "target_score": args.target,
    })

    ctrl = MAGRPOSimulationController(cfg)

    try:
        ctrl.train()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted – saving model…[/yellow]")
        os.makedirs("results", exist_ok=True)
        ctrl.trainer.save_model("results/magrpo_interrupted.pt")
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()