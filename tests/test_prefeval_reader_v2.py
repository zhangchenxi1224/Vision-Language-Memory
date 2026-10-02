import json
from pathlib import Path
import unittest
from scripts.eval.prefeval_rgb import load_overlay,mcq_score
from scripts.experiments.register_prefeval_reader_v2 import original_jobs
from vision_memory.prefeval.rgb_protocol import digest,queries_v2
from vision_memory.reader.open_answer import score_short_answer
ROOT=Path(__file__).resolve().parents[1]

class ReaderV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=ROOT/'reports/prefeval-rgb-20260917'
        cls.m=json.loads((p/'registered/manifest.json').read_text(encoding='utf-8'))
        cls.o=load_overlay(p/'reader-format-v2.json',cls.m)

    def test_no_relaxed_scoring(self):
        self.assertFalse(mcq_score('<B>',1)['strict_correct'])
        for raw,gold in [('scope: I avoid nuts.','I avoid nuts.'),('I avoid nuts.','"I avoid nuts."'),('not green','green')]:
            self.assertFalse(score_short_answer(raw,gold)['strict_correct'])

    def test_fixed_pairing(self):
        old=original_jobs(self.m);new=self.o['reference_jobs']
        self.assertEqual(len(new),1024)
        for a,b in zip(old,new,strict=True):
            self.assertEqual(a['id'],b['v1_id'])
            for k in ('state','target','target_index','split','groups','scope','form'):
                self.assertEqual(a.get(k),b.get(k))
        self.assertEqual(self.m['membership_sha'],'3cfd2e99dd34e7c828e1c60a75760052140ebc18220b85be9176e66309b8de0c')

    def test_quotation_contract_and_serialization(self):
        q=queries_v2({'topic':'"I avoid nuts."'},training=True)
        self.assertEqual(q[1]['target'],'"I avoid nuts."')
        self.assertNotIn('without quotation marks',q[1]['query'])
        self.assertIn('quotation marks already present',q[1]['query'])
        for job in self.o['reference_jobs']:
            for value in job['state'].values():
                if value is not None:self.assertIn('\n'+value+'\n',job['text_prefix'])

    def test_foils_and_supplement(self):
        self.assertLessEqual(len(self.o['supplement_jobs'])*2,1920)
        for t in self.o['teachers'].values():
            for q in t['qualification']:
                if q.get('foil_group'):self.assertIn(q['foil_group'],self.m['pilot_train'])
        self.assertEqual(set(self.o['teachers']),set(self.m['targets']))

    def test_baseline_not_deduplicated_by_state(self):
        eps=self.o['baseline']['episodes']
        self.assertEqual(sum(len(e['transitions']) for e in eps),124)
        self.assertEqual(len(eps),100)
        for ep in eps[96:]:
            self.assertEqual(ep['transitions'][-2]['state'],ep['transitions'][-1]['state'])
            self.assertNotEqual(ep['transitions'][-2]['ordinal'],ep['transitions'][-1]['ordinal'])

if __name__=='__main__':unittest.main()
