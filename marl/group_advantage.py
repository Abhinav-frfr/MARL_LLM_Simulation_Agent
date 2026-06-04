"""
marl/group_advantage.py - Corrected Implementation

Key fixes vs original:
1. Group axis is G (parallel trajectories) NOT separate agents
   Original confused agents with group members
2. Undiscounted Monte Carlo returns (paper uses sum not discounted sum)  
3. Removed GAE (not in paper)
4. Removed cooperation_bonus (not in paper)
5. Advantage per (g, t) pair matches Equation 1 exactly
"""

import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class GroupAdvantageCalculator:
    """
    Implements MAGRPO group-relative advantage calculation.
    
    Paper Equation 1:
        A_hat^(g)_t = R^(g)_t - (1/G) * sum_{g=1}^{G} R^(g)_t
    
    where R^(g)_t = sum_{tau=t}^{H-1} r^(g)_tau  (Monte Carlo return)
    
    CRITICAL CLARIFICATION vs original code:
    - g indexes PARALLEL TRAJECTORIES (group members), NOT agents
    - All n agents share the SAME joint reward r^(g)_tau
    - Advantage A_hat^(g)_t is the same for ALL agents in group g at step t
    - This is how centralized training works: joint signal trains all agents
    
    Paper: "we estimate the expected return of the current history
    by averaging over a group of Monte Carlo samples of the joint return"
    """

    def __init__(self):
        # No gamma needed - paper uses undiscounted returns
        # No gae_lambda - paper doesn't use GAE
        self.computation_log = []

    def compute_monte_carlo_returns(
        self,
        group_rewards: List[List[float]]
    ) -> List[List[float]]:
        """
        Algorithm 1 Line 11:
        R^(g)_t = sum_{tau=t}^{H-1} r^(g)_tau
        
        Undiscounted Monte Carlo return from step t to end of episode.
        Note: Paper shows NO discount factor gamma in this sum.
        
        Args:
            group_rewards: group_rewards[g][t] = joint reward at step t,
                           group g. Single scalar per (g,t) pair.
        
        Returns:
            returns[g][t] = R^(g)_t (undiscounted sum from t to H-1)
        """
        G = len(group_rewards)
        returns = []

        for g in range(G):
            H = len(group_rewards[g])
            returns_g = []
            running_sum = 0.0

            # Backward pass: R^(g)_t = r^(g)_t + R^(g)_{t+1}
            for t in reversed(range(H)):
                running_sum += group_rewards[g][t]
                returns_g.insert(0, running_sum)

            returns.append(returns_g)

        return returns  # returns[g][t]

    def compute_group_relative_advantages(
        self,
        returns: List[List[float]]
    ) -> List[List[float]]:
        """
        Algorithm 1 Line 12 / Equation 1:
        
        A_hat^(g)_t = R^(g)_t - (1/G) * sum_{g=1}^{G} R^(g)_t
        
        For each timestep t:
        1. Collect returns from ALL G trajectories: {R^(1)_t, ..., R^(G)_t}
        2. Compute group mean: mu_t = (1/G) * sum_g R^(g)_t
        3. Subtract mean: A_hat^(g)_t = R^(g)_t - mu_t
        
        This is the CENTRALIZED part of CTDE:
        - Uses returns from all G trajectories to form baseline
        - No separate value network required
        - Same advantage used for ALL agents (joint signal)
        
        Args:
            returns: returns[g][t] = R^(g)_t from compute_monte_carlo_returns
        
        Returns:
            advantages[g][t] = A_hat^(g)_t
        """
        G = len(returns)
        if G == 0:
            return []

        H = len(returns[0])

        # advantages[g][t] - initialize
        advantages = [[0.0] * H for _ in range(G)]

        for t in range(H):
            # Collect R^(g)_t for all g
            returns_at_t = [returns[g][t] for g in range(G)]

            # Group mean baseline: (1/G) * sum_g R^(g)_t
            group_mean = sum(returns_at_t) / G

            # A_hat^(g)_t = R^(g)_t - group_mean
            for g in range(G):
                advantages[g][t] = returns[g][t] - group_mean

        logger.debug(
            f"Advantages computed: G={G}, H={H}, "
            f"mean={np.mean(advantages):.4f}"
        )

        return advantages  # advantages[g][t]

    def compute_advantages_from_rewards(
        self,
        group_rewards: List[List[float]]
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Convenience method: compute returns AND advantages together.
        Implements Algorithm 1 lines 11-12.
        
        Args:
            group_rewards: group_rewards[g][t] = joint reward
        
        Returns:
            advantages[g][t] = A_hat^(g)_t  (Equation 1)
            returns[g][t]    = R^(g)_t       (Monte Carlo)
        """
        # Line 11: Monte Carlo returns
        returns = self.compute_monte_carlo_returns(group_rewards)

        # Line 12: Group-relative advantages (Equation 1)
        advantages = self.compute_group_relative_advantages(returns)

        return advantages, returns

    def get_statistics(
        self,
        advantages: List[List[float]],
        returns: List[List[float]]
    ) -> Dict[str, float]:
        """
        Compute statistics for logging.
        No normalization applied here - paper doesn't mention it.
        """
        all_adv = [
            advantages[g][t]
            for g in range(len(advantages))
            for t in range(len(advantages[g]))
        ]
        all_ret = [
            returns[g][t]
            for g in range(len(returns))
            for t in range(len(returns[g]))
        ]

        stats = {
            "G": len(advantages),
            "H": len(advantages[0]) if advantages else 0,
            "advantage_mean": float(np.mean(all_adv)) if all_adv else 0.0,
            "advantage_std": float(np.std(all_adv)) if all_adv else 0.0,
            "advantage_max": float(np.max(all_adv)) if all_adv else 0.0,
            "advantage_min": float(np.min(all_adv)) if all_adv else 0.0,
            "return_mean": float(np.mean(all_ret)) if all_ret else 0.0,
            "return_std": float(np.std(all_ret)) if all_ret else 0.0,
        }

        return stats