"""
marl/joint_policy.py - Corrected Implementation

Key fixes vs original:
1. Each agent has fully INDEPENDENT network (no shared encoder)
   Paper: "decentralized execution" - each pi_{theta_i} is separate
2. NO value heads - paper uses Monte Carlo returns, no value model
3. NO evaluate_actions (PPO method) - MAGRPO has no IS ratio
4. Group sampling added: G actions per agent per step
5. get_log_prob for policy update (Equation 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SingleAgentPolicy(nn.Module):
    """
    Policy network for ONE agent: pi_{theta_i}
    
    Paper: "A joint policy is a set of LOCAL policies pi_i,
    which condition on the local observation-action history h_{i,t}"
    
    Each agent has its OWN independent network.
    NO shared encoder - that would couple agents during execution,
    violating decentralized execution requirement.
    
    NO value head - paper uses Monte Carlo group returns as baseline,
    not a learned value function.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()

        # Fully independent network per agent
        # Paper: each pi_{theta_i} is separately parameterized
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns action logits"""
        return self.network(x)

    def sample_action(
        self,
        state: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[int, torch.Tensor]:
        """
        Sample one action from policy.
        Used in Algorithm 1 line 6 for group generation.
        
        Returns:
            action_idx: sampled action index
            log_prob:   log pi_{theta_i}(a | h) as tensor (kept for grad)
        """
        logits = self.forward(state) / max(temperature, 1e-8)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def get_log_prob(
        self,
        state: torch.Tensor,
        action_idx: int
    ) -> torch.Tensor:
        """
        Compute log pi_{theta_i}(a^(g)_{i,t} | h^G_{i,t})
        
        Used in Equation 2 policy update.
        
        IMPORTANT: NO importance sampling ratio here.
        Paper explicitly states MAGRPO has no IS ratio and no clipping.
        This is just the raw log probability under CURRENT policy.
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_tensor = torch.tensor(
            action_idx,
            dtype=torch.long,
            device=state.device
        )
        return dist.log_prob(action_tensor)

    def get_greedy_action(self, state: torch.Tensor) -> int:
        """
        Get deterministic (greedy) action for inference.
        argmax pi_{theta_i}(. | h)
        """
        with torch.no_grad():
            logits = self.forward(state)
            return torch.argmax(logits, dim=-1).item()


class JointPolicy(nn.Module):
    """
    Collection of n independent agent policies.
    
    Paper: pi* = {pi*_1, ..., pi*_n}
    
    This class manages all agent policies together for convenience
    but each pi_{theta_i} remains fully independent (decentralized).
    
    Key differences from original:
    - No shared encoder (agents are truly independent)
    - No value heads (paper uses Monte Carlo, not learned baseline)
    - No evaluate_actions PPO method
    - Adds group sampling (G trajectories per step)
    - Adds proper log_prob computation for Equation 2
    """

    # Shared action space for all agents
    ACTION_SPACE = [
        "SHAKE",
        "ADD_LIGHT",
        "ADD_NORMAL",
        "ADD_HEAVY",
        "SEEK_ADVICE"
    ]

    def __init__(
        self,
        state_dim: int = 64,
        hidden_dim: int = 256,
        num_agents: int = 5,
        action_dim: int = 5
    ):
        super().__init__()

        self.num_agents = num_agents
        self.action_dim = action_dim
        self.state_dim = state_dim

        # Each agent i has its own INDEPENDENT policy pi_{theta_i}
        # ModuleList so PyTorch tracks parameters correctly
        # NO shared encoder - that would violate decentralized execution
        self.agent_policies = nn.ModuleList([
            SingleAgentPolicy(state_dim, action_dim, hidden_dim)
            for _ in range(num_agents)
        ])

        logger.info(
            f"JointPolicy: {num_agents} independent agents, "
            f"state_dim={state_dim}, action_dim={action_dim}, "
            f"NO shared encoder, NO value heads"
        )

    def extract_features(self, state: Dict[str, Any]) -> torch.Tensor:
        """
        Convert state dict to fixed-size feature tensor.
        This is the numerical representation of observation o_{i,t}.
        """
        ball_counts = state.get('ball_counts', {})
        total_balls = max(sum(ball_counts.values()), 1)

        features = [
            # Current score
            float(state.get('score', 0.0)),
            # Best score so far this episode
            float(state.get('best_score', 0.0)),
            # Normalized steps without improvement
            float(state.get('steps_without_improvement', 0)) / 20.0,
            # Advice cooldown (normalized)
            float(state.get('advice_cooldown', 0)) / 3.0,
            # Ball distribution (raw counts normalized)
            float(ball_counts.get('LIGHT', 0)) / 100.0,
            float(ball_counts.get('NORMAL', 0)) / 100.0,
            float(ball_counts.get('HEAVY', 0)) / 100.0,
            # Ball distribution (proportions)
            float(ball_counts.get('LIGHT', 0)) / total_balls,
            float(ball_counts.get('NORMAL', 0)) / total_balls,
            float(ball_counts.get('HEAVY', 0)) / total_balls,
            # Score gap to target
            max(0.0, 0.7 - float(state.get('score', 0.0))),
        ]

        # Pad to state_dim with zeros
        while len(features) < self.state_dim:
            features.append(0.0)

        return torch.tensor(
            features[:self.state_dim],
            dtype=torch.float32
        )

    def generate_group_responses(
        self,
        agent_states: List[Dict[str, Any]],
        G: int,
        temperature: float = 1.0
    ) -> Tuple[List[List[int]], List[List[torch.Tensor]]]:
        """
        Algorithm 1 Line 6:
        Generate G responses for each agent i.
        
        a^G_{i,t} = {a^(1)_{i,t}, ..., a^(G)_{i,t}} ~ pi_{theta_i}(.|h^G_{i,t})
        
        Each agent independently samples G actions.
        Decentralized: agent i only uses its own state, not others'.
        
        Args:
            agent_states: List of state dicts, one per agent
            G:            Number of parallel trajectories (group size)
            temperature:  Sampling temperature
        
        Returns:
            group_actions[g][i]    = action index for agent i, group g
            group_log_probs[g][i]  = log prob tensor (with grad for update)
        """
        # Convert states to tensors
        state_tensors = [
            self.extract_features(s) for s in agent_states
        ]

        # group_actions[g][i], group_log_probs[g][i]
        group_actions: List[List[int]] = []
        group_log_probs: List[List[torch.Tensor]] = []

        for g in range(G):
            actions_g = []
            log_probs_g = []

            for i, (policy, state_tensor) in enumerate(
                zip(self.agent_policies, state_tensors)
            ):
                # Each agent i samples independently (decentralized)
                with torch.no_grad():
                    action_idx, log_prob = policy.sample_action(
                        state_tensor, temperature
                    )
                actions_g.append(action_idx)
                log_probs_g.append(log_prob)

            group_actions.append(actions_g)
            group_log_probs.append(log_probs_g)

        return group_actions, group_log_probs

    def compute_policy_gradient_loss(
        self,
        agent_idx: int,
        group_states: List[List[torch.Tensor]],
        group_actions: List[List[List[int]]],
        group_advantages: List[List[float]]
    ) -> torch.Tensor:
        """
        Equation 2 from paper:
        
        J(theta_i) = E_{o_0~D, h^G~pi_theta} [
            (1/G) * sum_g A_hat^(g)_t * log pi_{theta_i}(a^(g)_{i,t} | h^G_{i,t})
        ]
        
        NO importance sampling ratio (no pi_new / pi_old)
        NO epsilon clipping
        NO KL penalty (coefficient = 0)
        
        Args:
            agent_idx:        Index i of agent being updated
            group_states:     group_states[g][t] = state tensor at step t, group g
            group_actions:    group_actions[g][t][i] = action of agent i
            group_advantages: group_advantages[g][t] = A_hat^(g)_t
        
        Returns:
            loss: scalar tensor (negative J for gradient descent)
        """
        policy = self.agent_policies[agent_idx]
        G = len(group_states)
        H = len(group_states[0]) if G > 0 else 0

        total_objective = torch.tensor(0.0)

        for g in range(G):
            for t in range(H):
                state_tensor = group_states[g][t]
                action = group_actions[g][t][agent_idx]
                advantage = group_advantages[g][t]

                # log pi_{theta_i}(a^(g)_{i,t} | h^G_{i,t})
                # NO IS ratio - direct log prob under current policy
                log_prob = policy.get_log_prob(state_tensor, action)

                # Accumulate: A_hat^(g)_t * log pi_{theta_i}(...)
                total_objective = total_objective + advantage * log_prob

        # J(theta_i) = (1/G) * sum_g [...]
        # Minimize -J (gradient ascent on J)
        loss = -(1.0 / max(G, 1)) * total_objective

        return loss

    def get_action_name(self, action_idx: int) -> str:
        """Convert action index to string name"""
        if 0 <= action_idx < len(self.ACTION_SPACE):
            return self.ACTION_SPACE[action_idx]
        return "UNKNOWN"

    def get_action_index(self, action_name: str) -> int:
        """Convert action string to index"""
        try:
            return self.ACTION_SPACE.index(action_name)
        except ValueError:
            return 0  # Default to SHAKE

    def forward(
        self,
        state_tensors: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Forward pass: returns logits for each agent.
        Each agent processes only its own state (decentralized).
        
        Returns:
            logits[i] = action logits for agent i
        """
        logits = []
        for i, (policy, state) in enumerate(
            zip(self.agent_policies, state_tensors)
        ):
            logits.append(policy(state))
        return logits