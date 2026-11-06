import importlib
import importlib.util
import unittest


class BalancedRepeatTargetsTest(unittest.TestCase):
    def test_returns_only_the_four_preregistered_external_directions(self) -> None:
        module_name = "cell_level.multicenter.targets"
        spec = importlib.util.find_spec(module_name)
        self.assertIsNotNone(spec)
        targets = importlib.import_module(module_name)

        observed = {
            train_center: targets.external_test_centers(train_center)
            for train_center in ("bln", "bue", "vie14")
        }

        self.assertEqual(
            observed,
            {
                "bln": ("bue", "vie14"),
                "bue": ("bln",),
                "vie14": ("bue",),
            },
        )


if __name__ == "__main__":
    unittest.main()
