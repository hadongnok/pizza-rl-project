from __future__ import annotations

from typing import Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PizzaKitchenStage3Env(gym.Env):
    """
    Stage 3 Pizza Kitchen environment (FIFO Order Processing).

    Menu codes:
        0: empty order slot
        1: cheese pizza       (dough -> cheese)
        2: pepperoni pizza    (dough -> cheese -> pepperoni)
        3: mushroom pizza     (dough -> cheese -> mushroom)
        4: special pizza      (dough -> cheese -> pepperoni -> mushroom -> olive)

    Ingredient codes:
        0: dough
        1: cheese
        2: pepperoni
        3: mushroom
        4: olive

    Action codes:
        0: idle
        1: dough
        2: cheese
        3: pepperoni
        4: mushroom
        5: olive
        6: serve
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 2}

    ACTION_NAMES = (
        "idle",
        "dough",
        "cheese",
        "pepperoni",
        "mushroom",
        "olive",
        "serve",
    )

    INGREDIENT_NAMES = ("dough", "cheese", "pepperoni", "mushroom", "olive")

    # Menu code 0 is reserved for an empty order slot.
    RECIPES = {
        1: (0, 1),          # Cheese pizza
        2: (0, 1, 2),       # Pepperoni pizza
        3: (0, 1, 3),       # Mushroom pizza
        4: (0, 1, 2, 3, 4), # Special pizza
    }

    MENU_NAMES = {
        1: "cheese",
        2: "pepperoni",
        3: "mushroom",
        4: "special",
    }

    def __init__(
        self,
        n_orders: int = 4,
        min_deadline: int = 6,
        max_deadline: int = 16,
        observation_mode: str = "discrete",
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        if n_orders < 1:
            raise ValueError("n_orders must be at least 1")
        if min_deadline < 1 or max_deadline < min_deadline:
            raise ValueError("Invalid deadline range")
        if observation_mode not in ("discrete", "vector"):
            raise ValueError("observation_mode must be 'discrete' or 'vector'")
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError("Unsupported render_mode")

        self.n_orders = int(n_orders)
        self.min_deadline = int(min_deadline)
        self.max_deadline = int(max_deadline)
        self.observation_mode = observation_mode
        self.render_mode = render_mode

        # 5 ingredient flags + idle + 5 ingredient actions + serve = 7 actions.
        self.action_space = spaces.Discrete(7)

        # Five binary ingredient flags, then (menu, remaining_time) per order.
        self.radices = (2, 2, 2, 2, 2) + (
            (5, self.max_deadline + 1) * self.n_orders
        )
        self.n_states = int(np.prod(self.radices, dtype=np.int64))

        if observation_mode == "discrete":
            self.observation_space = spaces.Discrete(self.n_states)
        else:
            self.observation_space = spaces.Box(
                low=np.zeros(len(self.radices), dtype=np.float32),
                high=np.ones(len(self.radices), dtype=np.float32),
                dtype=np.float32,
            )

        self.plate = [0, 0, 0, 0, 0]
        self.orders: list[list[int]] = []
        self.steps = 0
        self.episode_return = 0.0
        self.stats: dict = {}

    def _raw_state(self) -> tuple[int, ...]:
        return tuple(self.plate) + tuple(
            value for order in self.orders for value in order
        )

    def encode_state(self, state: tuple[int, ...]) -> int:
        if len(state) != len(self.radices):
            raise ValueError("State length does not match configuration")

        result = 0
        for value, base in zip(state, self.radices):
            value = int(value)
            if not 0 <= value < base:
                raise ValueError(
                    f"State component {value} outside 0..{base - 1}"
                )
            result = result * base + value
        return result

    def decode_state(self, state_id: int) -> tuple[int, ...]:
        state_id = int(state_id)
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

        return np.asarray(
            [value / (base - 1) for value, base in zip(state, self.radices)],
            dtype=np.float32,
        )

    def _get_info(self, event: str = "reset") -> dict:
        return {
            "event": event,
            "raw_state": self._raw_state(),
            "steps": self.steps,
            "episode_return": self.episode_return,
            **self.stats.copy(),
        }

    def _plate_sequence(self) -> tuple[int, ...]:
        """Return added ingredient codes in recipe order."""
        return tuple(i for i, added in enumerate(self.plate) if added == 1)

    def _is_valid_ingredient(self, ingredient: int) -> bool:
        """
        An ingredient is valid only if adding it extends the current plate
        to a prefix of at least one defined recipe.
        """
        current = self._plate_sequence()
        candidate = current + (ingredient,)

        return any(
            len(candidate) <= len(recipe)
            and recipe[:len(candidate)] == candidate
            for recipe in self.RECIPES.values()
        )

    def _get_served_menu(self) -> int:
        """Return exact matching menu code, or 0 if the plate is not a recipe."""
        current = self._plate_sequence()
        for menu, recipe in self.RECIPES.items():
            if current == recipe:
                return menu
        return 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        self.plate = [0, 0, 0, 0, 0]
        self.steps = 0
        self.episode_return = 0.0
        self.stats = {
            "served": 0,
            "expired": 0,
            "wrong_serves": 0,
            "invalid_actions": 0,
            "idle_actions": 0,
        }

        fixed_orders = None if options is None else options.get("orders")

        if fixed_orders is not None:
            if len(fixed_orders) != self.n_orders:
                raise ValueError("Expected exactly n_orders fixed orders")

            self.orders = []
            for menu, deadline in fixed_orders:
                if menu not in self.RECIPES:
                    raise ValueError(f"Invalid menu code: {menu}")
                if not 1 <= deadline <= self.max_deadline:
                    raise ValueError(f"Invalid deadline: {deadline}")
                self.orders.append([int(menu), int(deadline)])
        else:
            self.orders = [
                [
                    int(self.np_random.integers(1, len(self.RECIPES) + 1)),
                    int(
                        self.np_random.integers(
                            self.min_deadline, self.max_deadline + 1
                        )
                    ),
                ]
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

        elif 1 <= action <= 5:
            ingredient = action - 1

            if self._is_valid_ingredient(ingredient):
                self.plate[ingredient] = 1
            else:
                reward -= 2.0
                self.stats["invalid_actions"] += 1
                event = "invalid_ingredient"

        else:  # action == 6: serve
            served_menu = self._get_served_menu()

            matching_orders = [
                i
                for i, (menu, _) in enumerate(self.orders)
                if menu != 0 and menu == served_menu
            ]

            if matching_orders:
                # FIFO 핵심 변경: 남은 마감 시간 정렬을 제거하고 가장 앞쪽 인덱스(가장 먼저 들어온 주문)를 선택
                slot = min(matching_orders)
                self.orders[slot] = [0, 0]
                self.stats["served"] += 1
                reward += 20.0
                event = "served"

            elif self._plate_sequence() == ():
                reward -= 2.0
                self.stats["invalid_actions"] += 1
                event = "empty_serve"

            else:
                reward -= 15.0
                self.stats["wrong_serves"] += 1
                event = "wrong_serve"

            self.plate = [0, 0, 0, 0, 0]

        for order in self.orders:
            if order[0] != 0:
                order[1] -= 1
                if order[1] == 0:
                    order[0] = 0
                    self.stats["expired"] += 1
                    reward -= 10.0

        terminated = all(menu == 0 for menu, _ in self.orders)
        truncated = False

        self.episode_return += reward
        info = self._get_info(event)

        if self.render_mode == "human":
            self.render()

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self):
        message = (
            f"Step {self.steps} | plate={self.plate} | "
            f"orders={self.orders} | return={self.episode_return:.1f}"
        )
        if self.render_mode == "human":
            print(message)
        return message
