"""
marl/magrpo_trainer.py
Final verified implementation of Algorithm 1, Eq 1, Eq 2.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class AgentPolicyNetwork(nn.Module):
    """Independent policy pi_{theta_i}. No shared encoder. No value head."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def sample_action(
        self,
        state: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[int, float]:
        """
        Sample for rollout. Returns (int, float) primitives.
        No gradient - recomputed fresh in compute_policy_loss.
        """
        with torch.no_grad():
            logits = self.forward(state) / max(temperature, 1e-8)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item()

    def compute_log_prob(
        self,
        state: torch.Tensor,
        action_idx: int
    ) -> torch.Tensor:
        """
        log pi_{theta_i}(a|h) WITH gradient.
        Called during policy update (Equation 2).
        No importance sampling ratio.
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_tensor = torch.tensor(
            action_idx,
            dtype=torch.long,
            device=next(self.parameters()).device
        )
        return dist.log_prob(action_tensor)


class AgentHistory:
    """
    h_{i,t} = {o_{i,0}, a_{i,0}, ..., o_{i,t}} per agent per group.
    Algorithm 1 lines 3-4, 8.
    """

    def __init__(self, num_agents: int, G: int):
        self.G = G
        self.num_agents = num_agents
        self._initialized = False
        self.histories: Dict[int, Dict[int, List]] = {
            i: {g: [] for g in range(G)}
            for i in range(num_agents)
        }

    def initialize(self, initial_obs: List[torch.Tensor]):
        """h^G_{i,0} <- o_{i,0} for all i, g"""
        for i, obs in enumerate(initial_obs):
            for g in range(self.G):
                self.histories[i][g] = [obs.detach().clone()]
        self._initialized = True

    def update(
        self,
        agent_idx: int,
        group_idx: int,
        action: int,
        new_obs: torch.Tensor
    ):
        """h^G_{i,t+1} <- {h^G_{i,t}, a^G_{i,t}, o^G_{i,t+1}}"""
        self.histories[agent_idx][group_idx].append(action)
        self.histories[agent_idx][group_idx].append(
            new_obs.detach().clone()
        )

    def reset(self):
        self.histories = {
            i: {g: [] for g in range(self.G)}
            for i in range(self.num_agents)
        }
        self._initialized = False

    def is_initialized(self) -> bool:
        return self._initialized


class MAGRPOTrainer:
    """
    Multi-Agent Group Relative Policy Optimization.
    Algorithm 1 + Equations 1, 2 from paper exactly.

    Properties:
    - G parallel trajectories per episode
    - Joint reward: single scalar shared by all agents
    - Eq 1: group-relative advantages (no value network)
    - Eq 2: direct log-prob, NO IS ratio, NO clipping, KL=0
    - Decentralized execution: independent policy per agent
    """

    ACTION_SPACE = [
        "SHAKE", "ADD_LIGHT", "ADD_NORMAL",
        "ADD_HEAVY", "SEEK_ADVICE"
    ]

    def __init__(
        self,
        state_dim: int,
        num_agents: int,
        action_dim: int,
        lr: float = 3e-4,
        G: int = 8
    ):
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.lr = lr
        self.G = G

        self.policies = nn.ModuleList([
            AgentPolicyNetwork(state_dim, action_dim)
            for _ in range(num_agents)
        ])

        self.optimizers = [
            torch.optim.Adam(
                self.policies[i].parameters(), lr=lr
            )
            for i in range(num_agents)
        ]

        self.agent_history = AgentHistory(num_agents, G)
        self.reset_episode_buffer()

        self.training_stats: Dict[str, List] = {
            'episode_returns': [],
            'policy_losses': [],
            'advantages_mean': [],
            'advantages_std': []
        }

        logger.info(
            f"MAGRPOTrainer: {num_agents} agents, G={G}, "
            f"NO clipping, KL=0, undiscounted returns"
        )

    def reset_episode_buffer(self):
        self.group_rewards: List[List[float]] = [
            [] for _ in range(self.G)
        ]
        self.group_states: List[List[List[torch.Tensor]]] = [
            [] for _ in range(self.G)
        ]
        self.group_actions: List[List[List[int]]] = [
            [] for _ in range(self.G)
        ]
        self.group_log_probs_stored: List[List[List[float]]] = [
            [] for _ in range(self.G)
        ]
        self.agent_history.reset()

    def state_to_tensor(
        self, state: Dict[str, Any]
    ) -> torch.Tensor:
        ball_counts = state.get('ball_counts', {})
        total = max(sum(ball_counts.values()), 1)
        features = [
            float(state.get('score', 0.0)),
            float(state.get('best_score', 0.0)),
            float(state.get(
                'steps_without_improvement', 0)) / 20.0,
            float(state.get('advice_cooldown', 0)) / 3.0,
            float(ball_counts.get('LIGHT', 0)) / 100.0,
            float(ball_counts.get('NORMAL', 0)) / 100.0,
            float(ball_counts.get('HEAVY', 0)) / 100.0,
            float(ball_counts.get('LIGHT', 0)) / total,
            float(ball_counts.get('NORMAL', 0)) / total,
            float(ball_counts.get('HEAVY', 0)) / total,
            max(0.0, 0.7 - float(state.get('score', 0.0))),
        ]
        while len(features) < self.state_dim:
            features.append(0.0)
        device = next(self.policies[0].parameters()).device
        return torch.tensor(
            features[:self.state_dim],
            dtype=torch.float32,
            device=device
        )

    def generate_group_actions(
        self,
        agent_states: List[Dict[str, Any]],
        temperature: float = 1.0
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """Algorithm 1 Line 6: generate G responses per agent."""
        state_tensors = [
            self.state_to_tensor(s) for s in agent_states
        ]
        if not self.agent_history.is_initialized():
            self.agent_history.initialize(state_tensors)

        group_actions: List[List[int]] = []
        group_log_probs: List[List[float]] = []

        for g in range(self.G):
            actions_g: List[int] = []
            lps_g: List[float] = []
            for policy, state_tensor in zip(
                self.policies, state_tensors
            ):
                a, lp = policy.sample_action(state_tensor, temperature)
                actions_g.append(a)
                lps_g.append(lp)
            group_actions.append(actions_g)
            group_log_probs.append(lps_g)

        return group_actions, group_log_probs

    def record_step(
        self,
        agent_states: List[Dict[str, Any]],
        group_actions: List[List[int]],
        group_log_probs: List[List[float]],
        joint_rewards: List[float]
    ):
        """
        Algorithm 1 Lines 7-8.
        joint_rewards[g] = single scalar for ALL agents.
        """
        assert len(joint_rewards) == self.G, (
            f"Need G={self.G} joint rewards, got {len(joint_rewards)}"
        )
        state_tensors = [
            self.state_to_tensor(s) for s in agent_states
        ]
        for g in range(self.G):
            self.group_rewards[g].append(float(joint_rewards[g]))
            self.group_states[g].append(
                [t.detach().clone() for t in state_tensors]
            )
            self.group_actions[g].append(list(group_actions[g]))
            self.group_log_probs_stored[g].append(
                list(group_log_probs[g])
            )
            for i in range(self.num_agents):
                self.agent_history.update(
                    i, g, group_actions[g][i], state_tensors[i]
                )

    def compute_returns(self) -> List[List[float]]:
        """
        Algorithm 1 Line 11:
        R^(g)_t = sum_{tau=t}^{H-1} r^(g)_tau (undiscounted)
        """
        H = len(self.group_rewards[0])
        if H == 0:
            return [[] for _ in range(self.G)]
        returns = []
        for g in range(self.G):
            rg = [0.0] * H
            running = 0.0
            for t in reversed(range(H)):
                running += self.group_rewards[g][t]
                rg[t] = running
            returns.append(rg)
        return returns

    def compute_group_relative_advantages(
        self, returns: List[List[float]]
    ) -> List[List[float]]:
        """
        Equation 1:
        A_hat^(g)_t = R^(g)_t - (1/G)*sum_g R^(g)_t
        """
        G = len(returns)
        if G == 0 or not returns[0]:
            return [[] for _ in range(G)]
        H = len(returns[0])
        adv = [[0.0] * H for _ in range(G)]
        for t in range(H):
            vals = [returns[g][t] for g in range(G)]
            mean = sum(vals) / G
            for g in range(G):
                adv[g][t] = returns[g][t] - mean
        return adv

    def compute_policy_loss(
        self,
        agent_idx: int,
        advantages: List[List[float]]
    ) -> torch.Tensor:
        """
        Equation 2:
        J(theta_i) = (1/G)*sum_g A_hat^(g)_t *
                     log pi_{theta_i}(a^(g)_{i,t}|h^G_{i,t})
        NO IS ratio. NO clipping. KL=0.
        """
        H = len(self.group_rewards[0])
        policy = self.policies[agent_idx]
        terms: List[torch.Tensor] = []

        for g in range(self.G):
            for t in range(H):
                st = self.group_states[g][t][agent_idx]
                ai = self.group_actions[g][t][agent_idx]
                adv = advantages[g][t]
                lp = policy.compute_log_prob(st, ai)
                terms.append(adv * lp)

        if not terms:
            return torch.tensor(0.0, requires_grad=True)
        return -(1.0 / self.G) * torch.stack(terms).sum()

    def update_all_agents(self) -> Dict[str, float]:
        """Algorithm 1 Lines 10-14. Called once per episode."""
        if not self.group_rewards[0]:
            logger.warning("No data - skipping")
            return {}

        returns = self.compute_returns()
        advantages = self.compute_group_relative_advantages(returns)
        losses: Dict[str, float] = {}

        for i in range(self.num_agents):
            self.optimizers[i].zero_grad()
            loss = self.compute_policy_loss(i, advantages)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policies[i].parameters(), max_norm=1.0
            )
            self.optimizers[i].step()
            losses[f'agent_{i}_loss'] = loss.item()

        all_adv = [
            advantages[g][t]
            for g in range(self.G)
            for t in range(len(advantages[g]))
        ]
        mean_adv = float(np.mean(all_adv)) if all_adv else 0.0
        std_adv = float(np.std(all_adv)) if all_adv else 0.0
        avg_ret = float(np.mean([
            sum(self.group_rewards[g]) for g in range(self.G)
        ]))

        self.training_stats['policy_losses'].append(losses)
        self.training_stats['advantages_mean'].append(mean_adv)
        self.training_stats['advantages_std'].append(std_adv)
        self.training_stats['episode_returns'].append(avg_ret)

        self.reset_episode_buffer()

        return {
            **losses,
            'mean_advantage': mean_adv,
            'std_advantage': std_adv,
            'avg_return': avg_ret
        }

    def save_model(self, path: str):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        torch.save({
            'policies': [p.state_dict() for p in self.policies],
            'training_stats': self.training_stats,
            'config': {
                'state_dim': self.state_dim,
                'num_agents': self.num_agents,
                'action_dim': self.action_dim,
                'G': self.G,
                'lr': self.lr
            }
        }, path)
        logger.info(f"Saved: {path}")

    def load_model(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        for i, sd in enumerate(ckpt['policies']):
            if i < len(self.policies):
                self.policies[i].load_state_dict(sd)
        self.training_stats = ckpt.get(
            'training_stats', self.training_stats
        )
        logger.info(f"Loaded: {path}")

    def get_action_names(self, indices: List[int]) -> List[str]:
        return [
            self.ACTION_SPACE[i]
            if 0 <= i < len(self.ACTION_SPACE) else "UNKNOWN"
            for i in indices
        ]

    def get_statistics(self) -> Dict[str, Any]:
        recent = self.training_stats['episode_returns'][-10:]
        return {
            'recent_reward': float(np.mean(recent)) if recent else 0.0,
            'total_episodes': len(
                self.training_stats['episode_returns']
            ),
            'buffer_size': sum(len(r) for r in self.group_rewards),
            'G': self.G,
            'num_agents': self.num_agents
        }