import unittest
import regressionmodel as rm
import numpy as np
from numpy.testing import assert_allclose


class TestRModel(unittest.TestCase):
    def test_rotate_simple(
        self,
    ):  # returns bool -> def rotate_point(x_cor, y_cor, theta, pivot_idx, move_idx):
        # Test should rotate in a complete circle
        x_cor = np.array([1.0, 2.0, 3.0], dtype=float)
        y_cor = np.array([3.0, 2.0, 1.0], dtype=float)
        theta = 180
        pivot_idx = 1
        move_idx = 2

        first_checkx = np.array([1.0, 2.0, 1.0], dtype=float)
        first_checky = np.array([3.0, 2.0, 3.0], dtype=float)
        second_checkx = np.array([1.0, 2.0, 1.0], dtype=float)
        second_checky = np.array([3.0, 2.0, 1.0], dtype=float)
        third_checkx = np.array([1.0, 2.0, 3.0], dtype=float)
        third_checky = np.array([3.0, 2.0, 1.0], dtype=float)

        rm.rotate_point(x_cor, y_cor, theta, pivot_idx, move_idx)
        assert_allclose(x_cor, first_checkx, atol=1e-15)
        assert_allclose(y_cor, first_checky, atol=1e-15)

        theta = 90
        rm.rotate_point(x_cor, y_cor, theta, pivot_idx, move_idx)
        assert_allclose(x_cor, second_checkx, atol=1e-15)
        assert_allclose(y_cor, second_checky, atol=1e-15)

        theta = 45
        rm.rotate_point(x_cor, y_cor, theta, pivot_idx, move_idx)
        rm.rotate_point(x_cor, y_cor, theta, pivot_idx, move_idx)
        assert_allclose(x_cor, third_checkx, atol=1e-15)
        assert_allclose(y_cor, third_checky, atol=1e-15)

        theta = -360
        rm.rotate_point(x_cor, y_cor, theta, pivot_idx, move_idx)
        assert_allclose(x_cor, third_checkx, atol=1e-15)
        assert_allclose(y_cor, third_checky, atol=1e-15)

    def test_shift_simple(
        self,
    ):  # returns bool -> def shift_points(x_cor, y_cor, x_samt, y_samt):
        x_cor = np.array([1.0, 2.0, 3.0], dtype=float)
        y_cor = np.array([3.0, 2.0, 1.0], dtype=float)

        first_checkx = np.array([2.0, 3.0, 4.0], dtype=float)
        first_checky = np.array([3.0, 2.0, 1.0], dtype=float)
        second_checkx = np.array([2.0, 3.0, 4.0], dtype=float)
        second_checky = np.array([4.0, 3.0, 2.0], dtype=float)
        third_checkx = np.array([1.0, 2.0, 3.0], dtype=float)
        third_checky = np.array([3.0, 2.0, 1.0], dtype=float)

        x_samt = 1
        y_samt = 0
        rm.shift_points(x_cor, y_cor, x_samt, y_samt)
        assert_allclose(x_cor, first_checkx, atol=1e-15)
        assert_allclose(y_cor, first_checky, atol=1e-15)
        x_samt = 0
        y_samt = 1
        rm.shift_points(x_cor, y_cor, x_samt, y_samt)
        assert_allclose(x_cor, second_checkx, atol=1e-15)
        assert_allclose(y_cor, second_checky, atol=1e-15)
        x_samt = -1
        y_samt = -1
        rm.shift_points(x_cor, y_cor, x_samt, y_samt)
        assert_allclose(x_cor, third_checkx, atol=1e-15)
        assert_allclose(y_cor, third_checky, atol=1e-15)

    def helper_distance(self, x_cor, y_cor):
        first_x_dif = x_cor[2] - x_cor[1]
        first_y_dif = y_cor[2] - y_cor[1]
        second_x_dif = x_cor[1] - x_cor[0]
        second_y_dif = y_cor[1] - y_cor[0]

        first_x_dif = first_x_dif * first_x_dif
        first_y_dif = first_y_dif * first_y_dif
        second_x_dif = second_x_dif * second_x_dif
        second_y_dif = second_y_dif * second_y_dif

        return np.sqrt(first_x_dif + first_y_dif), np.sqrt(second_x_dif + second_y_dif)

    def test_distance_between_pivot_and_outer(
        self,
    ):  # makes sure distance between points don't change
        grid_dimension = 1000
        # goal
        x_goal = np.random.randint(0, grid_dimension, 3).astype(float)
        y_goal = np.random.randint(0, grid_dimension, 3).astype(float)

        # user data
        x_cor = np.random.randint(0, grid_dimension, 3).astype(float)
        y_cor = np.random.randint(0, grid_dimension, 3).astype(float)

        og_dist1, og_dist2 = self.helper_distance(x_cor, y_cor)
        rm.run_regression(x_goal, y_goal, x_cor, y_cor, 1000, np.array([180, 180]))

        new_dist1, new_dist2 = self.helper_distance(x_cor, y_cor)

        assert_allclose(og_dist1, new_dist1, atol=1e-15)
        assert_allclose(og_dist2, new_dist2, atol=1e-15)


if __name__ == "__main__":

    unittest.main()
