"""Check device exclusion, registered coverage and failure cancellation."""
import importlib.util
from pathlib import Path
import threading
import unittest

spec = importlib.util.spec_from_file_location('scheduler', Path(__file__).parents[1] / 'scripts/inspire/run_prefeval_b730_exposure512_two_gpu.py')
scheduler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler)


class ScheduleTests(unittest.TestCase):
    def test_each_registered_evaluation_once_and_gpu_zero_reused_after_training(self):
        trained = threading.Event()
        calls = []
        class Core:
            def train(self):
                trained.set()
            def evaluate(self, step, split, variant, gpu):
                if gpu == 0:
                    assert trained.is_set()
                calls.append((step, split, variant, gpu))
            def terminate_children(self):
                raise AssertionError('Unexpected cancellation')
        scheduler.execute(Core())
        keys = [c[:3] for c in calls]
        self.assertEqual(len(keys), 19)
        self.assertEqual(set(keys), set(scheduler.registered_evaluations()))
        self.assertEqual({c[3] for c in calls}, {0, 1})
        for step, split, variant, gpu in calls:
            if step != 23360:
                self.assertEqual(variant, gpu)

    def test_failed_training_cancels_waiting_evaluation_lane(self):
        stopped = threading.Event()
        class Core:
            def train(self):
                raise RuntimeError('simulated training failure')
            def evaluate(self, *args):
                assert stopped.wait(5), 'Waiting lane was not cancelled'
                raise RuntimeError('cancelled')
            def terminate_children(self):
                stopped.set()
        with self.assertRaises(RuntimeError):
            scheduler.execute(Core())
        self.assertTrue(stopped.is_set())


if __name__ == '__main__':
    unittest.main()
