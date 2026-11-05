import random
import unittest

import numpy as np
import torch

from cell_level.ctransformer import train as stage1_train
from cell_level.pactransformer import train as stage2_train


def _draw_rng_triplet() -> tuple[float, float, float]:
    return random.random(), float(np.random.random()), float(torch.rand(1).item())


class TrainingSeedReproducibilityTest(unittest.TestCase):
    def test_stage1_seed_helper_reproduces_all_training_rngs(self) -> None:
        self.assertTrue(hasattr(stage1_train, "seed_everything"))
        seed_everything = stage1_train.seed_everything
        seed_everything(31415)
        first = _draw_rng_triplet()
        seed_everything(31415)
        second = _draw_rng_triplet()

        self.assertEqual(first, second)

    def test_stage2_seed_helper_reproduces_all_training_rngs(self) -> None:
        self.assertTrue(hasattr(stage2_train, "seed_everything"))
        seed_everything = stage2_train.seed_everything
        seed_everything(27182)
        first = _draw_rng_triplet()
        seed_everything(27182)
        second = _draw_rng_triplet()

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
