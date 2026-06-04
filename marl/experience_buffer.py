"""
marl/experience_buffer.py - Corrected Implementation

Key fixes vs original:
1. Stores G parallel trajectories per episode (not single trajectory)
2. Joint reward stored as single scalar (not list of per-agent rewards)
3. Buffer organized by groups matching Algorithm 1 structure
4. Removed individual reward tracking
5. get_group_trajectories returns correct (g,t) indexed structure
"""

from collections import deque
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EpisodeData:
    """
    Stores data for ONE episode with G parallel trajectories.
    
    Matches Algorithm 1 structure:
    - G groups, each with H steps
    - Each step has joint reward (single scalar for all agents)
    - Each step has group_actions[g][i] = action of agent i in group g
    """

    def __init__(self, G: int, num_agents: int):
        self.G = G
        self.num_agents = num_agents

        # group_rewards[g][t] = single joint reward for ALL agents
        # Paper: R: S^acc x A -> R  (single scalar)
        self.group_rewards: List[List[float]] = [[] for _ in range(G)]

        # group_actions[g][t][i] = action index of agent i at step t, group g
        self.group_actions: List[List[List[int]]] = [[] for _ in range(G)]

        # group_states[g][t][i] = state tensor of agent i at step t, group g
        self.group_states: List[List[List[Any]]] = [[] for _ in range(G)]

        # group_log_probs[g][t][i] = log prob of agent i action
        self.group_log_probs: List[List[List[Any]]] = [[] for _ in range(G)]

        self.episode_length = 0
        self.completed = False

    def add_step(
        self,
        group_actions: List[List[int]],
        group_log_probs: List[List[Any]],
        group_states: List[List[Any]],
        joint_rewards: List[float]
    ):
        """
        Add one timestep t across all G groups.
        
        Algorithm 1 lines 6-8:
        - group_actions[g][i] = a^(g)_{i,t}  (line 6)
        - joint_rewards[g]    = r^(g)_t       (line 7, single scalar)
        - group_states[g][i]  = state at t    (line 8)
        
        Args:
            group_actions:  [G][num_agents] action indices
            group_log_probs:[G][num_agents] log probs
            group_states:   [G][num_agents] state tensors
            joint_rewards:  [G] single scalar per group (NOT per agent)
        """
        assert len(joint_rewards) == self.G, (
            f"Need one joint reward per group. "
            f"Got {len(joint_rewards)}, expected {self.G}. "
            f"Joint reward is ONE scalar for ALL agents."
        )

        for g in range(self.G):
            # ONE joint reward for group g (shared by all agents)
            self.group_rewards[g].append(float(joint_rewards[g]))
            self.group_actions[g].append(group_actions[g])
            self.group_states[g].append(group_states[g])
            self.group_log_probs[g].append(group_log_probs[g])

        self.episode_length += 1

    def finalize(self):
        """Mark episode as complete"""
        self.completed = True

    def get_total_return(self, g: int) -> float:
        """Get total undiscounted return for group g: sum_{t=0}^{H-1} r^(g)_t"""
        return sum(self.group_rewards[g])

    def get_mean_return(self) -> float:
        """Get mean return across all G groups"""
        return float(np.mean([
            self.get_total_return(g) for g in range(self.G)
        ]))


class MARLExperienceBuffer:
    """
    Experience buffer for MAGRPO.
    
    Stores completed episodes, each with G parallel trajectories.
    
    Key differences from original:
    - Each episode stores G trajectories (not 1)
    - Joint reward is single scalar per (g,t) (not list per agent)
    - Buffer organized to match Algorithm 1 exactly
    - No individual reward storage
    
    Paper Algorithm 1:
    - Line 6: generate G responses per agent
    - Line 7: obtain joint rewards r^G_t (G scalars)
    - Lines 10-14: backward pass uses stored (g,t) indexed data
    """

    def __init__(
        self,
        capacity: int = 1000,
        num_agents: int = 5,
        G: int = 8
    ):
        """
        Args:
            capacity:   Max episodes to store
            num_agents: Number of agents n
            G:          Group size (parallel trajectories per episode)
        """
        self.capacity = capacity
        self.num_agents = num_agents
        self.G = G

        # Store completed episodes
        self.completed_episodes: deque = deque(maxlen=capacity)

        # Current episode being built
        self.current_episode: Optional[EpisodeData] = None

        self.episode_count = 0

        logger.info(
            f"MARLExperienceBuffer: capacity={capacity}, "
            f"G={G}, num_agents={num_agents}"
        )

    def start_episode(self):
        """
        Begin new episode.
        Algorithm 1 line 2: "Sample a task ~ D"
        """
        self.current_episode = EpisodeData(self.G, self.num_agents)
        logger.debug(f"Started episode {self.episode_count + 1}")

    def add_step(
        self,
        group_actions: List[List[int]],
        group_log_probs: List[List[Any]],
        group_states: List[List[Any]],
        joint_rewards: List[float]
    ):
        """
        Add one timestep to current episode.
        
        Algorithm 1 lines 6-8.
        
        CRITICAL: joint_rewards must have length G (one per group),
        NOT length num_agents. All agents share the same reward.
        
        Args:
            group_actions:  [G][num_agents] - actions for each agent in each group
            group_log_probs:[G][num_agents] - log probs
            group_states:   [G][num_agents] - states
            joint_rewards:  [G] - ONE scalar per group for ALL agents
        """
        if self.current_episode is None:
            self.start_episode()

        self.current_episode.add_step(
            group_actions=group_actions,
            group_log_probs=group_log_probs,
            group_states=group_states,
            joint_rewards=joint_rewards
        )

    def finish_episode(self) -> Optional[EpisodeData]:
        """
        Finalize current episode and add to buffer.
        Called after Algorithm 1 inner loop (line 9: end for).
        
        Returns the completed episode for immediate use in update.
        """
        if self.current_episode is None:
            logger.warning("finish_episode called with no active episode")
            return None

        self.current_episode.finalize()
        self.completed_episodes.append(self.current_episode)

        mean_return = self.current_episode.get_mean_return()
        ep_len = self.current_episode.episode_length

        logger.debug(
            f"Episode {self.episode_count} done: "
            f"length={ep_len}, mean_return={mean_return:.3f}"
        )

        completed = self.current_episode
        self.current_episode = None
        self.episode_count += 1

        return completed

    def get_latest_episode(self) -> Optional[EpisodeData]:
        """Get most recently completed episode for update"""
        if self.completed_episodes:
            return self.completed_episodes[-1]
        return None

    def get_recent_episodes(self, n: int = 10) -> List[EpisodeData]:
        """Get n most recent completed episodes"""
        episodes = list(self.completed_episodes)
        return episodes[-n:]

    def get_group_rewards(
        self,
        episode: EpisodeData
    ) -> List[List[float]]:
        """
        Extract group_rewards[g][t] from episode.
        Used in compute_monte_carlo_returns (Algorithm 1 line 11).
        
        Returns:
            group_rewards[g][t] = joint reward at step t, group g
        """
        return episode.group_rewards

    def get_group_states_and_actions(
        self,
        episode: EpisodeData
    ) -> Tuple[List[List[List[Any]]], List[List[List[int]]]]:
        """
        Extract states and actions for policy update.
        Used in compute_policy_gradient_loss (Algorithm 1 line 13).
        
        Returns:
            group_states[g][t][i]  = state of agent i at step t, group g
            group_actions[g][t][i] = action of agent i at step t, group g
        """
        return episode.group_states, episode.group_actions

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics for logging"""
        if not self.completed_episodes:
            return {
                "episodes_stored": 0,
                "total_episodes": self.episode_count,
                "G": self.G,
                "num_agents": self.num_agents
            }

        mean_returns = [
            ep.get_mean_return()
            for ep in self.completed_episodes
        ]
        ep_lengths = [
            ep.episode_length
            for ep in self.completed_episodes
        ]

        return {
            "episodes_stored": len(self.completed_episodes),
            "total_episodes": self.episode_count,
            "G": self.G,
            "num_agents": self.num_agents,
            "avg_mean_return": float(np.mean(mean_returns)),
            "best_mean_return": float(np.max(mean_returns)),
            "avg_episode_length": float(np.mean(ep_lengths)),
            "reward_type": "joint_scalar_per_group"
        }

    def clear(self):
        """Clear all stored episodes"""
        self.completed_episodes.clear()
        self.current_episode = None
        logger.info("Buffer cleared")