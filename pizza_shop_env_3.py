import gymnasium as gym
from gymnasium import spaces
import numpy as np

SHAPING_REWARD = 5.0
COMPLETION_SHAPING_REWARD = 10.0 

class PizzaShopEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.max_orders = 4
        self.max_steps = 100
        self.menus = {
            1: [1, 1, 0, 0, 0],  # Cheese
            2: [1, 1, 1, 0, 0],  # Pepperoni
            3: [1, 1, 0, 1, 0],  # Mushroom
            4: [1, 1, 1, 1, 1],  # Special
        }
        # 0 Idle, 1 Dough, 2 Cheese, 3 Pepperoni, 4 Mushroom, 5 Olive, 6 Serve
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=np.zeros(13, dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 4, 1, 4, 1, 4, 1, 4, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.counter = np.zeros(5, dtype=np.float32)
        self.orders = []
        self.steps = 0

    def _get_obs(self):
        obs = list(self.counter)
        for i in range(self.max_orders):
            if i < len(self.orders):
                obs.extend([self.orders[i]['menu'], self.orders[i]['timer'] / 48.0])
            else:
                obs.extend([0, 0])
        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.counter = np.zeros(5, dtype=np.float32)
        self.orders = [
            {'menu': int(self.np_random.choice([1, 2, 3, 4])),
             'timer': int(self.np_random.integers(36, 49))}
            for _ in range(self.max_orders)
        ]
        self.steps = 0
        return self._get_obs(), {}

    def _next_required_ingredient(self):
        if not self.orders:
            return None
        expected = self.menus[self.orders[0]['menu']]
        for idx, needed in enumerate(expected):
            if needed == 1 and self.counter[idx] == 0:
                return idx
        return None

    def step(self, action):
        action = int(action)
        reward = -0.1
        success_count = 0
        timeout_count = 0
        self.steps += 1
        next_required = self._next_required_ingredient()

        if action == 0:
            pass
        elif action == 1:  # Dough
            if self.counter[0] == 0:
                self.counter[0] = 1
                if next_required == 0:
                    reward += (COMPLETION_SHAPING_REWARD
                               if np.array_equal(self.counter, self.menus[self.orders[0]['menu']])
                               else SHAPING_REWARD)
            else:
                reward -= 2
        elif action == 2:  # Cheese
            if self.counter[0] == 1 and self.counter[1] == 0:
                self.counter[1] = 1
                if next_required == 1:
                    reward += (COMPLETION_SHAPING_REWARD
                               if np.array_equal(self.counter, self.menus[self.orders[0]['menu']])
                               else SHAPING_REWARD)
            else:
                reward -= 2
        elif action in (3, 4, 5):  # Toppings
            idx = action - 1
            if self.counter[0] == 1 and self.counter[1] == 1 and self.counter[idx] == 0:
                self.counter[idx] = 1
                if next_required == idx:
                    reward += (COMPLETION_SHAPING_REWARD
                               if np.array_equal(self.counter, self.menus[self.orders[0]['menu']])
                               else SHAPING_REWARD)
            else:
                reward -= 2
        elif action == 6:  # Serve (FIFO)
            if self.orders:
                expected = self.menus[self.orders[0]['menu']]
                if np.array_equal(self.counter, expected):
                    reward += 150
                    success_count += 1
                    self.orders.pop(0)
                else:
                    reward -= 15
                self.counter = np.zeros(5, dtype=np.float32)

        # Every active order loses one step, including on a serve action.
        for order in self.orders:
            order['timer'] -= 1
        remaining = []
        for order in self.orders:
            if order['timer'] <= 0:
                reward -= 50
                timeout_count += 1
            else:
                remaining.append(order)
        self.orders = remaining

        terminated = len(self.orders) == 0
        truncated = self.steps >= self.max_steps and not terminated
        info = {'success_count': success_count, 'timeout_count': timeout_count}
        return self._get_obs(), reward, terminated, truncated, info
