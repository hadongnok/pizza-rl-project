import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PizzaShopEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.max_orders = 4 

        # [d, c, p, m1, t1, m2, t2, m3, t3, m4, t4] = 11 dim
        # d, c, p: 0 or 1
        # m1, m2, m3, m4: 0 (No order), 1 (Cheese), 2 (Pepperoni)
        # t1, t2, t3, t4: 0 to 24 (Normalized by 24.0)
        # Menu IDs can reach 2, so the upper bound is not 1 for all dimensions.
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.array(
                [1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1],
                dtype=np.float32
            ),
            shape=(11,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(5)
        # 0: Idle, 1: Add Dough, 2: Add Cheese,
        # 3: Add Pepperoni, 4: Serve

        self.menus = {
            1: [1.0, 1.0, 0.0],  # Cheese Pizza
            2: [1.0, 1.0, 1.0]   # Pepperoni Pizza
        }

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.counter = np.zeros(3, dtype=np.float32)
        self.orders = []

        for _ in range(self.max_orders):
            self._add_order()

        self.steps = 0
        self.terminated_count = 0
        return self._get_obs(), {}

    def _add_order(self):
        if len(self.orders) < self.max_orders:
            menu_id = int(self.np_random.choice([1, 2]))
            timer = int(self.np_random.integers(16, 25)) 
            self.orders.append({
                'menu_id': menu_id,
                'timer': timer
            })

    def _get_obs(self):
        obs = list(self.counter)

        for i in range(self.max_orders):
            if i < len(self.orders):
                obs.append(float(self.orders[i]['menu_id']))
                obs.append(float(self.orders[i]['timer']) / 24.0) 
            else:
                obs.append(0.0)
                obs.append(0.0)

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        reward = -0.1
        terminated = False
        truncated = self.steps >= 100

        # 1. Action Processing & Rule Checking
        if action == 1:  # Add Dough
            if self.counter[0] == 1.0:
                reward -= 2.0
            else:
                self.counter[0] = 1.0

        elif action == 2:  # Add Cheese
            if self.counter[1] == 1.0:
                reward -= 2.0
            elif self.counter[0] == 0.0:
                reward -= 2.0
            else:
                self.counter[1] = 1.0

        elif action == 3:  # Add Pepperoni
            if self.counter[2] == 1.0:
                reward -= 2.0
            elif self.counter[0] == 0.0 or self.counter[1] == 0.0:
                reward -= 2.0
            else:
                self.counter[2] = 1.0

        elif action == 4:  # Serve
            if len(self.orders) > 0:
                target_menu = self.orders[0]['menu_id']
                target_recipe = self.menus[target_menu]

                if np.array_equal(self.counter, target_recipe):
                    reward += 20.0
                    self.orders.pop(0)
                    self.counter = np.zeros(3, dtype=np.float32)
                else:
                    reward -= 15.0
                    self.counter = np.zeros(3, dtype=np.float32)
            else:
                reward -= 1.0

        # 2. Timer Update & Timeout Check
        orders_to_remove = []

        for order in self.orders:
            order['timer'] -= 1.0
            if order['timer'] <= 0:
                reward -= 10.0
                orders_to_remove.append(order)

        for expired in orders_to_remove:
            if expired in self.orders:
                self.orders.remove(expired)

        if len(self.orders) == 0:
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}
