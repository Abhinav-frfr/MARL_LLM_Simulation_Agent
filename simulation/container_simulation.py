"""
simulation/container_simulation.py
"""

import random
import numpy as np
from typing import List, Tuple, Dict
import logging

from config.prompts import BALL_SYMBOLS

logger = logging.getLogger(__name__)


class ContainerSimulation:

    def __init__(self, size: int = 10):
        self.size = size
        self.container = [
            [0] * size for _ in range(size)
        ]
        self.action_history = []
        self.homogeneity_history = []

    def reset(self):
        self.container = [
            [0] * self.size for _ in range(self.size)
        ]
        self.action_history = []
        self.homogeneity_history = []

    # ----------------------------------------------------------
    # Ball Management
    # ----------------------------------------------------------

    def add_ball(
        self,
        ball_type: int,
        position: Tuple[int, int] = None
    ) -> bool:
        if position is None:
            empty = [
                (i, j)
                for i in range(self.size)
                for j in range(self.size)
                if self.container[i][j] == 0
            ]
            if not empty:
                return False
            pos = random.choice(empty)
            self.container[pos[0]][pos[1]] = ball_type
        else:
            if self.container[position[0]][position[1]] == 0:
                self.container[position[0]][position[1]] = ball_type
            else:
                return False
        return True

    def add_multiple_balls(self, ball_type: int, count: int):
        for _ in range(count):
            if not self.add_ball(ball_type):
                break

    # ----------------------------------------------------------
    # Physics
    # ----------------------------------------------------------

    def shake(self, duration: int = 10):
        num_swaps = duration * (self.size ** 2) // 20
        for _ in range(num_swaps):
            i1 = random.randint(0, self.size - 1)
            j1 = random.randint(0, self.size - 1)
            direction = random.choice(
                ['up', 'down', 'left', 'right']
            )
            i2, j2 = i1, j1

            if direction == 'up' and i1 > 0:
                i2 = i1 - 1
            elif direction == 'down' and i1 < self.size - 1:
                i2 = i1 + 1
            elif direction == 'left' and j1 > 0:
                j2 = j1 - 1
            elif direction == 'right' and j1 < self.size - 1:
                j2 = j1 + 1
            else:
                continue

            b1 = self.container[i1][j1]
            b2 = self.container[i2][j2]

            if b1 > 0 and b2 > 0:
                if b1 > b2 and direction == 'down':
                    prob = 0.7
                elif b2 > b1 and direction == 'up':
                    prob = 0.7
                else:
                    prob = 0.3
                if random.random() < prob:
                    self.container[i1][j1] = b2
                    self.container[i2][j2] = b1
            elif b1 > 0 and b2 == 0:
                if random.random() < 0.8:
                    self.container[i1][j1] = 0
                    self.container[i2][j2] = b1
            elif b2 > 0 and b1 == 0:
                if random.random() < 0.8:
                    self.container[i1][j1] = b2
                    self.container[i2][j2] = 0

    # ----------------------------------------------------------
    # Scoring
    # ----------------------------------------------------------

    def calculate_homogeneity(self) -> float:
        if self._get_ball_count() == 0:
            return 0.0

        total_diversity = 0.0
        neighbor_count = 0

        for i in range(self.size):
            for j in range(self.size):
                if self.container[i][j] > 0:
                    neighbors = []
                    for di, dj in [
                        (-1, 0), (1, 0), (0, -1), (0, 1)
                    ]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.size and 0 <= nj < self.size:
                            neighbors.append(self.container[ni][nj])
                    unique = set(n for n in neighbors if n > 0)
                    if unique:
                        total_diversity += len(unique) / 3.0
                        neighbor_count += 1

        if neighbor_count == 0:
            return 0.0
        return total_diversity / neighbor_count

    def get_joint_reward(
        self,
        score_before: float,
        score_after: float,
        target_score: float = 0.7,
        steps_without_improvement: int = 0
    ) -> float:
        """
        Single scalar joint reward for ALL agents.
        Paper: R: S^acc x A -> R
        """
        delta = score_after - score_before
        reward = delta
        if score_after >= target_score:
            reward += 1.0
        if steps_without_improvement > 5:
            reward -= 0.05
        return float(reward)

    # ----------------------------------------------------------
    # State Access
    # ----------------------------------------------------------

    def get_state(self) -> List[List[int]]:
        return self.container

    def get_ball_distribution(self) -> Dict[str, int]:
        dist = {"LIGHT": 0, "NORMAL": 0, "HEAVY": 0}
        for row in self.container:
            for cell in row:
                if cell == 1:
                    dist["LIGHT"] += 1
                elif cell == 2:
                    dist["NORMAL"] += 1
                elif cell == 3:
                    dist["HEAVY"] += 1
        return dist

    def _get_ball_count(self) -> int:
        return sum(
            1 for row in self.container
            for c in row if c > 0
        )

    # ----------------------------------------------------------
    # State Copy/Restore (for G group simulation)
    # ----------------------------------------------------------

    def copy_state(self) -> List[List[int]]:
        return [row[:] for row in self.container]

    def restore_state(self, state: List[List[int]]):
        self.container = [row[:] for row in state]

    # ----------------------------------------------------------
    # Action Execution
    # ----------------------------------------------------------

    def execute_action(
        self, action: str, parameters: Dict
    ) -> Tuple[bool, str]:
        try:
            if action == "ADD_LIGHT":
                count = parameters.get("count", 1)
                self.add_multiple_balls(1, count)
                return True, f"Added {count} light ball(s)"
            elif action == "ADD_NORMAL":
                count = parameters.get("count", 1)
                self.add_multiple_balls(2, count)
                return True, f"Added {count} normal ball(s)"
            elif action == "ADD_HEAVY":
                count = parameters.get("count", 1)
                self.add_multiple_balls(3, count)
                return True, f"Added {count} heavy ball(s)"
            elif action == "SHAKE":
                duration = parameters.get("duration", 10)
                self.shake(duration)
                return True, f"Shook for {duration}s"
            elif action == "RESET":
                self.reset()
                return True, "Reset"
            else:
                return False, f"Unknown: {action}"
        except Exception as e:
            return False, f"Error: {e}"

    # ----------------------------------------------------------
    # Display
    # ----------------------------------------------------------

    def display(self):
        print("\n" + "=" * 50)
        print("CONTAINER STATE")
        print("=" * 50)
        for i, row in enumerate(self.container):
            row_str = " ".join(
                BALL_SYMBOLS.get(c, '?') for c in row
            )
            print(f"{i:2} | {row_str}")
        print("-" * 50)
        print(f"Score: {self.calculate_homogeneity():.3f}")
        print(f"Dist:  {self.get_ball_distribution()}")
        print("=" * 50)