"""Check device exclusion, registered coverage and failure cancellation."""
import importlib.util
from pathlib import Path
import threading
import tempfile
import sys
import time
import unittest

spec = importlib.util.spec_from_file_location('scheduler', Path(__file__).parents[1] / 'scripts/inspire/run_prefeval_b730_exposure512_two_gpu.py')
scheduler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler)


class ScheduleTests(unittest.TestCase):
    @unittest.skipIf(sys.platform == 'win32', 'Production orchestration requires POSIX')
    def test_four_plus_two_covers_same_registration_with_one_training_lane(self):
        sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts/inspire'))
        import run_prefeval_b730_exposure512_distributed as distributed
        calls, training = [], []
        class Core:
            def train(self):
                training.append('train')
            def evaluate(self, *args):
                calls.append(args)
        for role, count in [('primary', 4), ('reader', 2)]:
            for gpu in range(count):
                distributed.lane(Core(), role, gpu)
        self.assertEqual(training, ['train'])
        self.assertEqual(len(calls), 19)
        self.assertEqual({c[:3] for c in calls}, set(scheduler.registered_evaluations()))

    @unittest.skipIf(sys.platform == 'win32', 'Shared POSIX lock is exercised on Linux')
    def test_shared_endpoint_lock_prevents_duplicate_work(self):
        with tempfile.TemporaryDirectory() as directory:
            writes = []
            class Core:
                RUN = Path(directory)
                STOP = threading.Event()
                def evaluate(self, *args):
                    marker = self.RUN / 'summary.json'
                    if marker.exists():
                        return
                    time.sleep(0.1)
                    writes.append(args)
                    marker.write_text('{}')
            first, second = Core(), Core()
            scheduler.lock_evaluations(first)
            scheduler.lock_evaluations(second)
            with scheduler.concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                jobs = [pool.submit(c.evaluate, 93440, 'pilot', 0, gpu) for gpu, c in enumerate((first, second))]
                for job in jobs:
                    job.result()
            self.assertEqual(len(writes), 1)

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
