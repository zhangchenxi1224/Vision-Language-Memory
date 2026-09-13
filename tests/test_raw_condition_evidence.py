"""Integrity tests using an existing endpoint as a clearly synthetic raw arm."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.reporting.collect_broader_raw_condition import collect, PHASE, PROBE, CHECKPOINT, RUNTIME
from scripts.reporting.collect_broader_endpoint import phase_summary, BANK_SHA
from scripts.reporting.collect_transition_endpoint import read, jsonl, sha
from scripts.reporting.verify_broader_outputs_local import unpack
from scripts.experiments.broader_writer_protocol import SEED

ROOT = Path(__file__).resolve().parents[1]


class RawConditionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='synthetic-raw-evidence-')
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.parent, self.run = root/'parent', root/'synthetic-raw'
        unpack(ROOT/'reports/official-alignment-results-20260913/16bc3d0-logical-endpoint-evidence.tgz',
            'aea5d68397532008b873a9e5d6c3ed7b1dc5cf40167740081b5a096c20287230', self.parent)
        self.bank = self.parent/'bank/manifest.json'
        # Archive stores the bound bank under its run-local bank directory.
        self.phase = self.run/PHASE
        self.phase.mkdir(parents=True)
        for name in ('summary.json', 'complete.json'):
            shutil.copyfile(self.parent/'train/trained'/name, self.phase/name)
        self.rows = jsonl(self.parent/'train/trained/generations.jsonl')
        for row in self.rows:
            row['phase'] = PHASE
        identity = {'probe_commit': PROBE, 'checkpoint_sha256': CHECKPOINT,
            'parent_runtime_sha256': RUNTIME, 'parent_result_sha256': sha(self.parent/'train/result.json'),
            'bank_sha256': BANK_SHA, 'optimizer_updates': 0, 'native_steps': 28,
            'guidance_scale': 1., 'image_guidance_scale': 1., 'seed': SEED, 'eval_seeds': 2,
            'groups': 151, 'raw_rows': 3020, 'matched_rows': 1510,
            'condition_style': 'cached upstream raw-event training embedding and mask, repeated into the three native branches'}
        self.write(self.run/'identity.json', identity)
        native, _ = phase_summary(jsonl(self.parent/'train/trained/generations.jsonl'), read(self.bank), 'trained')
        raw = {**copy.deepcopy(native), 'phase': PHASE}
        self.complete = {'identity': identity, 'phase': PHASE, 'native_summary': native,
            'raw_summary': raw, 'matched_changes': {'1->1': 1460, '0->0': 50}, 'development_all_correct_eos': False}
        self.reseal()

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, sort_keys=True)+'\n', encoding='utf-8')

    def reseal(self):
        path = self.phase/'generations.jsonl'
        path.write_text(''.join(json.dumps(row)+'\n' for row in self.rows), encoding='utf-8')
        phase = read(self.phase/'complete.json')
        phase['artifact_hashes']['generations.jsonl'] = sha(path)
        self.write(self.phase/'complete.json', phase)
        self.complete['phase_complete_sha256'] = sha(self.phase/'complete.json')
        self.write(self.run/'complete.json', self.complete)

    def verify(self):
        return collect(self.run, self.parent, self.bank, text_only=True)

    def test_full_paired_recount(self):
        result = self.verify()
        self.assertEqual(result['raw_rows_recounted'], 6040)
        self.assertEqual(result['matched_changes'], {'1->1': 1460, '0->0': 50})
        self.assertEqual(len(result['artifacts_omitted_locally']), 605)

    def test_missing_cell_rejected_even_with_new_seal(self):
        self.rows.pop()
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.verify()

    def test_score_forgery_rejected_even_with_new_seal(self):
        row = next(row for row in self.rows if row['condition'] == 'matched' and row['scorer']['strict_correct'])
        row['generated_token_ids'] = [151645]
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'strict score'):
            self.verify()

    def test_changed_control_pixels_rejected(self):
        selected = self.rows[0]['question_id']
        for row in self.rows:
            if row['question_id'] == selected and row['condition'] == 'blank':
                row['image_sha256'] = '0'*64
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'negative-control'):
            self.verify()

    def test_changed_identity_rejected(self):
        identity = read(self.run/'identity.json')
        identity['optimizer_updates'] = 1
        self.write(self.run/'identity.json', identity)
        self.complete['identity'] = identity
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'inference control'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
