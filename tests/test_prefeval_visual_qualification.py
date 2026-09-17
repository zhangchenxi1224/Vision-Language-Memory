import copy
import json
from pathlib import Path
import unittest
from vision_memory.prefeval.rgb_protocol import recovery_summary,queries_v2
from scripts.experiments.prefeval_visual_policy import load_policy

class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.state={'a':None,'b':'I avoid nuts.'}
        self.queries=queries_v2(self.state)+[dict(kind='active_status',scope='b',query='active?',target='yes')]
        self.rows=[dict(query=q,png_sha256='png',score=dict(strict_correct=q['kind']=='recovery')) for q in self.queries]

    def test_recovery_and_auxiliary_have_distinct_meanings(self):
        s=recovery_summary(self.state,self.queries,self.rows,'png')
        self.assertTrue(s['recovery_complete']);self.assertFalse(s['all_registered_checks_pass'])

    def test_missing_duplicate_and_wrong_png_rejected(self):
        for rows in (self.rows[:-2]+self.rows[-1:],self.rows+[self.rows[0]],
                     [dict(r,png_sha256='wrong') for r in self.rows]):
            with self.assertRaises(ValueError):recovery_summary(self.state,self.queries,rows,'png')

    def test_cleared_and_untouched_errors_fail_complete_state(self):
        for index in (0,2):
            rows=copy.deepcopy(self.rows);rows[index]['score']['strict_correct']=False
            self.assertFalse(recovery_summary(self.state,self.queries,rows,'png')['recovery_complete'])

    def test_policy_keeps_historical_gate_failed_and_binds_overlay(self):
        root=Path(__file__).resolve().parents[1]/'reports/prefeval-rgb-20260917'
        m=json.loads((root/'registered/manifest.json').read_text(encoding='utf-8'))
        o=json.loads((root/'reader-format-v2.json').read_text(encoding='utf-8'))
        load_policy(root/'visual-recovery-allocation-v1.json',m,o)
        self.assertFalse(json.loads((root/'references-v2-verified.json').read_text())['gate_passed'])
        o['generation']['max_new_tokens']=256
        with self.assertRaises(ValueError):load_policy(root/'visual-recovery-allocation-v1.json',m,o)

if __name__=='__main__':unittest.main()
