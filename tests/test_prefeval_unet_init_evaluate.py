"""The evaluator must reject stale images and incomplete or mislabeled comparisons."""
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiments import prefeval_unet_init_evaluate as ev


def formatter(options):
    return '\nOPTIONS:' + json.dumps(options)


def extract(raw):
    match = re.search(r'<choice>([ABCD])</choice>', raw)
    return match.group(1) if match else None


MCQ = {'get_mcq_question_format': formatter, 'extract_choice': extract}


def fixture():
    rows = [{'base_pair_id': f'topic:{i:04d}', 'topic': 'topic',
             'forms': {f: f'{f} question {i}' for f in ev.FAMILIES},
             'options': [f'answer-{i}-{j}' for j in range(4)]} for i in range(2)]
    hashes = {(r['base_pair_id'], c): f"hash-{r['base_pair_id']}-{c}" for r in rows for c in range(2)}
    donors = ev.donor_map(rows)
    primary, positions = [], []
    for row in rows:
        pid = row['base_pair_id']
        for control in ev.CONTROLS:
            for chain in (range(2) if control in ('memory', 'mismatch') else range(1)):
                donor = donors[pid] if control == 'mismatch' else None
                digest = hashes[donor or pid, chain] if control in ('memory', 'mismatch') else None
                for family in ev.FAMILIES:
                    step = int.from_bytes(hashlib.sha256(f'eval:{pid}:{family}'.encode()).digest()[:4], 'big')
                    order, correct = ev.option_order(pid, step)
                    letter = 'ABCD'[correct]
                    primary.append({'pair_id': pid, 'chain': chain, 'prefix': 0, 'control': control,
                        'family': family, 'task': 'mcq', 'split': 'dev', 'endpoint_kind': 'student',
                        'max_new_tokens': 32, 'option_order': order, 'question': row['forms'][family],
                        'reader_query': row['forms'][family] + formatter([row['options'][i] for i in order]),
                        'donor_pair_id': donor, 'png_sha256': digest, 'correct_letter': letter,
                        'predicted_letter': letter, 'correct': True, 'parse_failure': False,
                        'generated': {'raw': f'<choice>{letter}</choice>', 'truncated': False}})
        for control in ev.POSITION_CONTROLS:
            for chain in (range(1) if control == 'blank' else range(2)):
                donor = donors[pid] if control == 'mismatch' else None
                digest = hashes[donor or pid, chain] if control != 'blank' else None
                for position in ev.POSITIONS:
                    order = ev.cyclic_order(position)
                    letter = 'ABCD'[order.index(0)]
                    positions.append({'pair_id': pid, 'chain': chain, 'prefix': 0, 'control': control,
                        'family': 'T1', 'position': position, 'order_mode': 'official-cyclic',
                        'split': 'dev', 'endpoint_kind': 'student', 'max_new_tokens': 32, 'order': order,
                        'reader_query': row['forms']['T1'] + formatter([row['options'][i] for i in order]),
                        'donor_pair_id': donor, 'png_sha256': digest, 'correct_letter': letter,
                        'predicted_letter': letter, 'correct': True, 'parse_failure': False,
                        'generated': {'raw': f'<choice>{letter}</choice>', 'truncated': False}})
    return rows, hashes, primary, positions


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.hashes, self.primary, self.positions = fixture()

    def validate(self, records=None, positions=False):
        return ev.validate_records(records if records is not None else self.primary, self.rows,
                                   self.hashes, 'dev', 2, MCQ, positions=positions)

    def test_complete_matrices_preserve_distinct_denominators(self):
        primary = self.validate()
        positions = self.validate(self.positions, positions=True)
        self.assertEqual((len(primary), len(positions)), (36, 50))
        summary = ev.aggregate(primary, positions, self.rows, 2)
        self.assertEqual(summary['families']['T1']['controls']['memory']['total'], 4)
        self.assertEqual(summary['families']['T1']['controls']['blank']['total'], 2)
        self.assertEqual(summary['positions']['memory']['all_four_cyclic_correct_per_image'], {'correct': 4, 'total': 4})
        self.assertEqual(summary['positions']['blank']['all_four_cyclic_correct_per_image'], {'correct': 2, 'total': 2})

    def test_missing_or_duplicated_read_is_never_complete(self):
        for records in (self.primary[:-1], self.primary + [self.primary[0]]):
            with self.subTest(length=len(records)), self.assertRaises(AssertionError):
                self.validate(records)
        with self.assertRaises(AssertionError):
            self.validate(self.positions[:-1], positions=True)

    def test_wrong_donor_or_stale_png_rejected(self):
        for field, value in [('donor_pair_id', 'topic:9999'), ('png_sha256', 'stale-image')]:
            records = copy.deepcopy(self.primary)
            next(r for r in records if r['control'] == 'mismatch')[field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                self.validate(records)

    def test_question_leak_and_changed_gold_rejected(self):
        for field, value in [('reader_query', 'Question including hidden preference'), ('correct_letter', 'Z')]:
            records = copy.deepcopy(self.primary)
            records[0][field] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                self.validate(records)

    def test_cyclic_positions_follow_semantic_option_zero(self):
        self.assertEqual([ev.cyclic_order(p).index(0) for p in range(4)], [0, 1, 2, 3])
        records = copy.deepcopy(self.positions)
        next(r for r in records if r['position'] == 1)['order'] = [0, 1, 2, 3]
        with self.assertRaises(AssertionError):
            self.validate(records, positions=True)

    def test_repair_and_joint_metrics_do_not_pool_questions_as_preferences(self):
        rec = next(r for r in self.primary if r['control'] == 'memory' and r['family'] == 'T2' and r['chain'] == 1)
        wrong = next(x for x in 'ABCD' if x != rec['correct_letter'])
        rec.update(correct=False, predicted_letter=wrong, generated={'raw': f'<choice>{wrong}</choice>', 'truncated': False})
        summary = ev.aggregate(self.validate(), self.validate(self.positions, positions=True), self.rows, 2)
        self.assertEqual(summary['families']['T2']['both_noise_correct'], {'correct': 1, 'total': 2})
        self.assertEqual(summary['all_three_questions_and_both_noise_correct'], {'correct': 1, 'total': 2})
        self.assertEqual(summary['families']['T2']['paired_memory_vs_mismatch'], {'repaired': 0, 'regressed': 1, 'total': 4})
        self.assertEqual(summary['families']['T2']['memory_minus_mismatch_pp'], -25)

    def test_parser_disagreement_cannot_be_counted_as_success(self):
        self.primary[0]['generated']['raw'] = 'not a valid choice'
        with self.assertRaises(AssertionError):
            self.validate()

    def test_cli_official_uses_history_without_initial_variants(self):
        args = ev.parser().parse_args(['--checkpoint', 'checkpoint.pt', '--output', 'out', '--base', 'base',
            '--official-source', 'source', '--reader', 'reader', '--split', 'official', '--history-file', 'history.json',
            '--expected-step', '23360'])
        self.assertEqual(args.split, 'official')
        self.assertIsNone(args.initial_variants)
        self.assertEqual(args.history_file, Path('history.json'))

    def test_completion_receipt_binds_endpoint_counts_and_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binding = {'checkpoint_sha256': 'checkpoint-digest', 'optimizer_step': 2048, 'split': 'dev'}
            summary = {'preferences': 90, 'primary_record_count': 1620, 'position_record_count': 2250,
                       'actual_png_count': 180}
            ev.save(root / 'binding.json', binding)
            ev.save(root / 'summary.json', summary)
            receipt = ev.complete_receipt(binding, root / 'binding.json', root / 'summary.json', summary)
            self.assertEqual(receipt['checkpoint_sha256'], 'checkpoint-digest')
            self.assertEqual((receipt['optimizer_step'], receipt['split'], receipt['actual_pngs']), (2048, 'dev', 180))
            ev.save(root / 'summary.json', dict(summary, actual_png_count=179))
            changed = ev.complete_receipt(binding, root / 'binding.json', root / 'summary.json', summary)
            self.assertNotEqual(changed['summary_sha256'], receipt['summary_sha256'])

    def test_real_png_metadata_seed_and_event_are_verified(self):
        from PIL import Image
        from vision_memory.training.latent_bank_unet import stable_seed
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = copy.deepcopy(self.rows)
            for row in rows:
                row['history'] = [{'role': 'user', 'content': 'I like apples.'},
                                  {'role': 'assistant', 'content': 'Noted.'}]
            variants = root / 'variants.json'
            ev.save(variants, {'review_complete': True, 'items': {
                row['base_pair_id']: {'preference': 'I like apples.',
                                     'variants': ['Noted.', 'Understood.', 'Thanks for telling me.']} for row in rows}})
            args = SimpleNamespace(output=root, split='dev', noise_chains=2, initial_variant=1, initial_variants=variants)
            binding = {'checkpoint_sha256': 'checkpoint', 'initial_variants_sha256': ev.sha(variants)}
            manifest = {'checkpoint_sha256': 'checkpoint', 'initial_variants_sha256': ev.sha(variants),
                        'initial_variant': 1, 'split': 'dev', 'steps': 28, 'cfg': 1, 'noise_chains': 2,
                        'inter_turns': 0, 'state': 'only reopened uint8 RGB PNG; fresh Gaussian each write'}
            ev.save(root / 'images/manifest.json', manifest)
            for row in rows:
                pid = row['base_pair_id']
                for chain in range(2):
                    directory = root / 'images' / pid.replace(':', '_') / f'seed-{chain}'
                    directory.mkdir(parents=True)
                    png = directory / 'prefix-00.png'
                    Image.new('RGB', (1024, 1024), (128, 127, 126)).save(png)
                    digest = ev.sha(png)
                    ev.save(directory / 'complete.json', {'binding': manifest, 'png_hashes': {'prefix-00.png': digest}})
                    write = {'position': 0, 'source_png_sha256': None, 'output_png_sha256': digest,
                             'noise_seed': stable_seed(20260924, f'rollout:{pid}:{chain}', 0),
                             'event': 'user: I like apples.\nassistant: Understood.'}
                    (directory / 'writes.jsonl').write_text(json.dumps(write) + '\n', encoding='utf-8')
            self.assertEqual(len(ev.validate_images(args, rows, binding)), 4)
            original = (directory / 'writes.jsonl').read_text()
            for field, value in [('noise_seed', 123), ('event', 'user: leaked question'), ('source_png_sha256', 'old-source')]:
                changed = dict(write, **{field: value})
                (directory / 'writes.jsonl').write_text(json.dumps(changed) + '\n', encoding='utf-8')
                with self.subTest(field=field), self.assertRaises(AssertionError):
                    ev.validate_images(args, rows, binding)
            (directory / 'writes.jsonl').write_text(original, encoding='utf-8')
            Image.new('RGB', (1024, 1024), (1, 2, 3)).save(png)
            with self.assertRaises(AssertionError):
                ev.validate_images(args, rows, binding)


if __name__ == '__main__':
    unittest.main()
