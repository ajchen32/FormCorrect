import unittest
import recursiveregressionmodel as rrm
import numpy as np
from numpy.testing import assert_allclose
import copy


class TestRecurse(unittest.TestCase):
    def test_3D_rotate_origin(
        self,
    ):

        rotate_point = rrm.Point(1, 0, 0)
        pivot_point = rrm.Point(0, 0, 0)

        # will compute the dot_product and determine the angle to test
        line_a = np.array(
            [
                pivot_point.x - rotate_point.x,
                pivot_point.y - rotate_point.y,
                pivot_point.z - rotate_point.z,
            ]
        )
        rrm.rotate_point_3D(rotate_point, pivot_point, 90, 0, 0)
        line_b = np.array(
            [
                pivot_point.x - rotate_point.x,
                pivot_point.y - rotate_point.y,
                pivot_point.z - rotate_point.z,
            ]
        )

        angle = self.helper_find_angle(line_a, line_b)
        self.assertAlmostEqual(angle, 0, delta=1e-15)
        rrm.rotate_point_3D(rotate_point, pivot_point, 0, 90, 0)
        line_b = np.array(
            [
                pivot_point.x - rotate_point.x,
                pivot_point.y - rotate_point.y,
                pivot_point.z - rotate_point.z,
            ]
        )
        angle = self.helper_find_angle(line_a, line_b)
        self.assertAlmostEqual(angle, 90, delta=1e-15)
        rrm.rotate_point_3D(rotate_point, pivot_point, 0, 0, 90)
        line_b = np.array(
            [
                pivot_point.x - rotate_point.x,
                pivot_point.y - rotate_point.y,
                pivot_point.z - rotate_point.z,
            ]
        )
        angle = self.helper_find_angle(line_a, line_b)
        self.assertAlmostEqual(angle, 90, delta=1e-15)

    def test_3D_rotate(
        self,
    ):
        # def rotate_point_3D(rotate_point, pivot_point, theta_x, theta_y, theta_z):
        first_point = np.random.randint(0, 10, 3).astype(float)
        second_point = np.random.randint(0, 10, 3).astype(float)

        rotate_point = rrm.Point(first_point[0], first_point[1], first_point[2])
        pivot_point = rrm.Point(second_point[0], second_point[1], second_point[2])

        # will compute the dot_product and determine the angle to test
        line_a = np.array(
            [
                pivot_point.x - rotate_point.x,
                pivot_point.y - rotate_point.y,
                pivot_point.z - rotate_point.z,
            ]
        )
        rrm.rotate_point_3D(rotate_point, pivot_point, 90, 0, 0)
        line_b = np.array(
            [
                pivot_point.x - rotate_point.x,
                pivot_point.y - rotate_point.y,
                pivot_point.z - rotate_point.z,
            ]
        )

        angle = self.helper_find_angle(line_a, line_b)
        # self.assertAlmostEqual(angle, 90, delta=1e-15)
        rrm.rotate_point_3D(rotate_point, pivot_point, 0, 90, 0)
        line_b = np.array(
            [
                pivot_point.x - rotate_point.x,
                pivot_point.y - rotate_point.y,
                pivot_point.z - rotate_point.z,
            ]
        )
        angle = self.helper_find_angle(line_a, line_b)
        # self.assertAlmostEqual(angle, 90, delta=1e-15)
        rrm.rotate_point_3D(rotate_point, pivot_point, 0, 0, 90)
        line_b = np.array(
            [
                pivot_point.x - rotate_point.x,
                pivot_point.y - rotate_point.y,
                pivot_point.z - rotate_point.z,
            ]
        )
        angle = self.helper_find_angle(line_a, line_b)
        # self.assertAlmostEqual(angle, 90, delta=1e-15)

    def helper_find_angle(self, line_a, line_b):  # returns the angle in degrees

        dot_prod = np.dot(line_a, line_b)
        norm_a = np.linalg.norm(line_a)
        norm_b = np.linalg.norm(line_b)

        radian_out = np.arccos((dot_prod / (norm_a * norm_b)))
        return np.degrees(radian_out)


if __name__ == "__main__":
    unittest.main()
