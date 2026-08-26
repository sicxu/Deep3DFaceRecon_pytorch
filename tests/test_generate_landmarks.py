import unittest

import numpy as np

from generate_landmarks import yunet_to_deep3d_landmarks


class LandmarkGeneratorTests(unittest.TestCase):
    def test_yunet_to_deep3d_landmarks(self):
        detection = np.array(
            [
                10, 20, 100, 120,  # bbox
                30, 40,            # right eye / image-left
                70, 41,            # left eye / image-right
                50, 60,            # nose
                35, 80,            # right mouth / image-left
                65, 81,            # left mouth / image-right
                0.99,              # confidence
            ],
            dtype=np.float32,
        )

        actual = yunet_to_deep3d_landmarks(detection)

        expected = np.array(
            [
                [30, 40],
                [70, 41],
                [50, 60],
                [35, 80],
                [65, 81],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
