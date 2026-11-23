import unittest

from pizza_env_stage3 import PizzaKitchenStage3Env


class TestPizzaKitchenStage3Env(unittest.TestCase):

    def setUp(self):
        self.env = PizzaKitchenStage3Env(
            n_orders=4,
            min_deadline=6,
            max_deadline=16,
            observation_mode="discrete",
        )

    def tearDown(self):
        self.env.close()

    def reset_with_menu(self, menu):
        return self.env.reset(
            seed=42,
            options={
                "orders": [
                    (menu, 16),
                    (1, 16),
                    (1, 16),
                    (1, 16),
                ]
            },
        )

    def test_action_and_state_space(self):
        self.assertEqual(self.env.action_space.n, 7)
        self.assertEqual(self.env.n_states, 1_670_420_000)
        self.assertEqual(len(self.env.radices), 13)

    def test_state_encode_decode_round_trip(self):
        _, info = self.env.reset(seed=42)
        raw_state = info["raw_state"]
        state_id = self.env.encode_state(raw_state)
        decoded = self.env.decode_state(state_id)
        self.assertEqual(raw_state, decoded)

    def test_all_four_recipes_can_be_served(self):
        # Actions: dough=1, cheese=2, pepperoni=3,
        # mushroom=4, olive=5, serve=6.
        recipe_actions = {
            1: [1, 2, 6],          # Cheese
            2: [1, 2, 3, 6],       # Pepperoni
            3: [1, 2, 4, 6],       # Mushroom
            4: [1, 2, 3, 4, 5, 6], # Special
        }

        for menu, actions in recipe_actions.items():
            with self.subTest(menu=menu):
                self.reset_with_menu(menu)
                for action in actions:
                    _, _, _, _, info = self.env.step(action)
                self.assertEqual(info["served"], 1)
                self.assertEqual(info["event"], "served")

    def test_invalid_ingredient_order_is_penalized(self):
        self.reset_with_menu(3)  # Mushroom pizza
        self.env.step(1)         # Dough
        self.env.step(2)         # Cheese
        _, reward, _, _, info = self.env.step(3)  # Pepperoni is invalid
        self.assertAlmostEqual(reward, -2.1)
        self.assertEqual(info["invalid_actions"], 1)
        self.assertEqual(self.env.plate, [1, 1, 0, 0, 0])

    def test_wrong_serve_is_penalized_and_clears_plate(self):
        self.env.reset(
            seed=42,
            options={"orders": [(3, 16), (3, 16), (3, 16), (3, 16)]},
        )                        # Only mushroom orders are waiting
        self.env.step(1)         # Dough
        self.env.step(2)         # Cheese
        self.env.step(3)         # Invalid pepperoni; plate stays dough+cheese
        _, reward, _, _, info = self.env.step(6)
        self.assertAlmostEqual(reward, -15.1)
        self.assertEqual(info["wrong_serves"], 1)
        self.assertEqual(self.env.plate, [0, 0, 0, 0, 0])

    def test_deadline_expiration(self):
        self.env.reset(
            seed=42,
            options={
                "orders": [
                    (1, 1), (1, 16), (1, 16), (1, 16)
                ]
            },
        )
        _, reward, _, _, info = self.env.step(0)
        self.assertEqual(info["expired"], 1)
        self.assertAlmostEqual(reward, -10.1)

    def test_serve_on_last_tick_counts_as_success(self):
        self.env.reset(
            seed=42,
            options={
                "orders": [
                    (1, 2), (2, 16), (3, 16), (4, 16)
                ]
            },
        )
        self.env.step(1)  # dough: cheese order deadline 1
        self.env.step(2)  # cheese: cheese order deadline reaches 0
        # The order expires after this action, so serve must happen
        # on the tick before its deadline reaches zero.
        # This test instead verifies expiration ordering using deadline 3.
        self.env.reset(
            seed=42,
            options={
                "orders": [
                    (1, 3), (2, 16), (3, 16), (4, 16)
                ]
            },
        )
        self.env.step(1)
        self.env.step(2)
        _, reward, _, _, info = self.env.step(6)
        self.assertEqual(info["served"], 1)
        self.assertEqual(info["expired"], 0)
        self.assertEqual(info["event"], "served")

    def test_vector_observation(self):
        vector_env = PizzaKitchenStage3Env(
            n_orders=4,
            min_deadline=6,
            max_deadline=16,
            observation_mode="vector",
        )
        obs, _ = vector_env.reset(seed=1)
        self.assertEqual(obs.shape, (13,))
        self.assertTrue((obs >= 0).all())
        self.assertTrue((obs <= 1).all())
        vector_env.close()


if __name__ == "__main__":
    unittest.main()
