"""Small deterministic tests for the calculus and forecast alignment."""
import unittest
import numpy as np
import pandas as pd

from src.model import ManualLinearRegressionGD, prepare_data

class GradientTests(unittest.TestCase):
    def test_analytical_gradient_matches_finite_difference(self):
        X = np.array([[1.0, 2.0], [2.0, -1.0], [0.5, 1.5]])
        y = np.array([2.0, -1.0, 1.0])
        w = np.array([0.3, -0.2])
        b = 0.1
        n = len(y)

        def loss(weights, bias):
            return np.mean((X @ weights + bias - y) ** 2)

        err = X @ w + b - y
        grad_w = (2 / n) * (X.T @ err)
        grad_b = 2 * np.mean(err)
        eps = 1e-6
        numeric_w = np.zeros_like(w)
        for j in range(len(w)):
            wp, wm = w.copy(), w.copy()
            wp[j] += eps; wm[j] -= eps
            numeric_w[j] = (loss(wp, b) - loss(wm, b)) / (2 * eps)
        numeric_b = (loss(w, b + eps) - loss(w, b - eps)) / (2 * eps)
        np.testing.assert_allclose(grad_w, numeric_w, rtol=1e-5, atol=1e-6)
        self.assertAlmostEqual(grad_b, numeric_b, places=5)

    def test_gradient_descent_reduces_training_loss(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        y = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        model = ManualLinearRegressionGD(learning_rate=0.03, iterations=300)
        model.fit(X, y)
        self.assertLess(model.loss_history[-1], model.loss_history[0])
        np.testing.assert_allclose(model.predict(X), y, atol=0.05)

    def test_real_data_alignment(self):
        # Four ten-minute timestamps: each feature timestamp must pair with t+10 min.
        dates = pd.date_range("2024-01-01 00:00", periods=4, freq="10min")
        df = pd.DataFrame({
            "date": dates, "Appliances": [10, 20, 30, 40], "lights": [1, 1, 1, 1],
            "T1": [20.0]*4, "RH_1": [40.0]*4, "T2": [20.0]*4, "RH_2": [40.0]*4,
            "T_out": [10.0]*4, "RH_out": [50.0]*4, "Windspeed": [2.0]*4,
        })
        prepared = prepare_data(df)
        self.assertEqual(len(prepared), 3)
        self.assertEqual(prepared["target_next_10min_Wh"].tolist(), [20, 30, 40])
        self.assertTrue(((pd.to_datetime(prepared["target_time"]) - pd.to_datetime(prepared["feature_time"])) == pd.Timedelta(minutes=10)).all())

if __name__ == "__main__":
    unittest.main()
