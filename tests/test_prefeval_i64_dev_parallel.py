import json
from pathlib import Path
import tempfile
import unittest

from scripts.inspire.prefeval_i64_dev_parallel import partition, save, sha, validate_shards


class DevPartitionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.bank = Path(cls.temp.name)
        cls.rows = [{'base_pair_id': f'item:{i:04d}'} for i in range(90)]
        cls.binding = {'checkpoint_sha256': 'checkpoint', 'split': 'dev', 'steps': 28, 'cfg': 1,
            'noise_chains': 2, 'inter_turns': 10,
            'state': 'only reopened uint8 RGB PNG; fresh Gaussian each write'}
        for shard in range(3):
            root = cls.bank / f'shard-{shard}'
            save(cls.bank / f'complete-{shard}.json', {'pairs': 30, 'chains': 60, 'shard': shard})
            save(root / 'manifest.json', cls.binding)
            for row in partition(cls.rows, shard):
                for chain in range(2):
                    parent = root / row['base_pair_id'].replace(':', '_') / f'seed-{chain}'
                    parent.mkdir(parents=True)
                    hashes, writes = {}, []
                    for position in range(11):
                        name = f'prefix-{position:02d}.png'
                        (parent / name).write_bytes(f'{row}:{chain}:{position}'.encode())
                        hashes[name] = sha(parent / name)
                        writes.append({'position': position, 'output_png_sha256': hashes[name],
                            'source_png_sha256': hashes[f'prefix-{position-1:02d}.png'] if position else None})
                    save(parent / 'complete.json', {'binding': cls.binding, 'png_hashes': hashes})
                    (parent / 'writes.jsonl').write_text('\n'.join(map(json.dumps, writes)))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_complete_disjoint_population_and_full_chain_hashes(self):
        parts = [partition(self.rows, i) for i in range(3)]
        self.assertEqual([len(p) for p in parts], [30, 30, 30])
        self.assertEqual({r['base_pair_id'] for p in parts for r in p}, {r['base_pair_id'] for r in self.rows})
        binding, paths = validate_shards(self.bank, self.rows, 'checkpoint')
        self.assertEqual(binding, self.binding)
        self.assertEqual(len(paths), 90)

    def test_corrupt_png_cannot_be_published(self):
        image = self.bank / 'shard-0/item_0000/seed-0/prefix-05.png'
        previous = image.read_bytes()
        try:
            image.write_bytes(b'corrupt')
            with self.assertRaises(AssertionError):
                validate_shards(self.bank, self.rows, 'checkpoint')
        finally:
            image.write_bytes(previous)

    def test_wrong_parent_cannot_be_published(self):
        with self.assertRaises(AssertionError):
            validate_shards(self.bank, self.rows, 'different checkpoint')


if __name__ == '__main__':
    unittest.main()
