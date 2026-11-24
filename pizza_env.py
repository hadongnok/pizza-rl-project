from __future__ import annotations

from typing import Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PizzaKitchenEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 2}
    ACTION_NAMES = ("idle", "dough", "cheese", "pepperoni", "serve")
    # Ingredient codes: dough=0, cheese=1, pepperoni=2
    RECIPES = {1: (0, 1), 2: (0, 1, 2)}

    def __init__(self, n_orders: int = 2, min_deadline: int = 4,
                 max_deadline: int = 8, observation_mode: str = "discrete",
                 render_mode: Optional[str] = None):
        super().__init__()
        if n_orders < 1 or min_deadline < 1 or max_deadline < min_deadline:
            raise ValueError("Invalid order count or deadline range")
        if observation_mode not in ("discrete", "vector"):
            raise ValueError("observation_mode must be 'discrete' or 'vector'")
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError("Unsupported render_mode")
        self.n_orders = int(n_orders)
        self.min_deadline = int(min_deadline)
        self.max_deadline = int(max_deadline)
        self.observation_mode = observation_mode
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(5)
        # 3 ingredient flags followed by (menu, remaining time) per order.
        self.radices = (2, 2, 2) + (3, self.max_deadline + 1) * self.n_orders
        self.n_states = int(np.prod(self.radices, dtype=np.int64))
        if observation_mode == "discrete":
            self.observation_space = spaces.Discrete(self.n_states)
        else:
            self.observation_space = spaces.Box(
                low=np.zeros(len(self.radices), dtype=np.float32),
                high=np.ones(len(self.radices), dtype=np.float32),
                dtype=np.float32,
            )
        self.plate = [0, 0, 0]
        self.orders = []
        self.steps = 0
        self.episode_return = 0.0
        self.stats = {}

    def _raw_state(self) -> tuple[int, ...]:
        return tuple(self.plate) + tuple(v for order in self.orders for v in order)

    def encode_state(self, state: tuple[int, ...]) -> int:
        if len(state) != len(self.radices):
            raise ValueError("State length does not match configuration")
        result = 0
        for value, base in zip(state, self.radices):
            if not 0 <= int(value) < base:
                raise ValueError(f"State component {value} outside 0..{base-1}")
            result = result * base + int(value)
        return result

    def decode_state(self, state_id: int) -> tuple[int, ...]:
        if not 0 <= state_id < self.n_states:
            raise ValueError("State ID outside observation space")
        values = []
        for base in reversed(self.radices):
            values.append(state_id % base)
            state_id //= base
        return tuple(reversed(values))

    def _get_obs(self):
        state = self._raw_state()
        if self.observation_mode == "discrete":
            return self.encode_state(state)
        return np.asarray([v / (base - 1) for v, base in zip(state, self.radices)],
                          dtype=np.float32)

    def _get_info(self, event: str = "reset") -> dict:
        return {
            "event": event,
            "raw_state": self._raw_state(),
            "steps": self.steps,
            "episode_return": self.episode_return,
            **self.stats.copy(),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.plate = [0, 0, 0]
        self.steps = 0
        self.episode_return = 0.0
        self.stats = {"served": 0, "expired": 0, "wrong_serves": 0,
                      "invalid_actions": 0, "idle_actions": 0}
        # options={'orders': [(menu, deadline), ...]} for deterministic tests.
        fixed = None if options is None else options.get("orders")
        if fixed is not None:
            if len(fixed) != self.n_orders:
                raise ValueError("Expected exactly n_orders fixed orders")
            self.orders = []
            for menu, deadline in fixed:
                if menu not in (1, 2) or not 1 <= deadline <= self.max_deadline:
                    raise ValueError("Invalid fixed menu/deadline")
                self.orders.append([int(menu), int(deadline)])
        else:
            self.orders = [
                [int(self.np_random.integers(1, 3)),
                 int(self.np_random.integers(self.min_deadline, self.max_deadline + 1))]
                for _ in range(self.n_orders)
            ]
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        if all(menu == 0 for menu, _ in self.orders):
            raise RuntimeError("Episode is finished. Call reset() first.")
        action = int(action)
        reward = -0.1
        event = self.ACTION_NAMES[action]
        self.steps += 1
        if action == 0:
            self.stats["idle_actions"] += 1
        elif action in (1, 2, 3):
            ingredient = action - 1
            # Enforce strict ingredient order and prohibit duplicates.
            valid = (ingredient == 0 and self.plate == [0, 0, 0]) or (
                ingredient == 1 and self.plate == [1, 0, 0]) or (
                ingredient == 2 and self.plate == [1, 1, 0])
            if valid:
                self.plate[ingredient] = 1
            else:
                reward -= 2.0
                self.stats["invalid_actions"] += 1
                event = "invalid_ingredient"
        else:  # bake and serve
            served_menu = 1 if self.plate == [1, 1, 0] else (
                2 if self.plate == [1, 1, 1] else 0)
            matching = [i for i, (menu, _) in enumerate(self.orders)
                        if menu != 0 and menu == served_menu]
            if matching:
                slot = min(matching)
                self.orders[slot] = [0, 0]
                self.stats["served"] += 1
                reward += 20.0
                event = "served"
            elif self.plate == [0, 0, 0]:
                reward -= 2.0  # empty plate is invalid, not a wasted pizza
                self.stats["invalid_actions"] += 1
                event = "empty_serve"
            else:
                reward -= 15.0
                self.stats["wrong_serves"] += 1
                event = "wrong_serve"
            self.plate = [0, 0, 0]
        # Serve at the last available tick counts as success.
        for order in self.orders:
            if order[0] != 0:
                order[1] -= 1
                if order[1] == 0:
                    order[0] = 0
                    self.stats["expired"] += 1
                    reward -= 10.0
        terminated = all(menu == 0 for menu, _ in self.orders)
        truncated = False  # All initial deadlines are finite; no new arrivals.
        self.episode_return += reward
        info = self._get_info(event)
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        message = (f"Step {self.steps} | plate={self.plate} | "
                   f"orders={self.orders} | return={self.episode_return:.1f}")
        if self.render_mode == "human":
            print(message)
        return message
