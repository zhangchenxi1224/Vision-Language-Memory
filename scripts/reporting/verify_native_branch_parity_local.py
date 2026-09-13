"""Recount every portable branch-audit cell after independent remote tensor checks."""
import argparse
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.reporting.collect_transition_endpoint import read,jsonl,sha
from scripts.reporting.verify_broader_outputs_local import unpack
from scripts.probes.native_training_branch_parity import NEW_RUNTIME_SHA,FIRST_STEP_COMPLETE_SHA,TOLERANCE


def verify(archive,digest):
    with tempfile.TemporaryDirectory(prefix='branch-parity-local-',dir=ROOT/'.cache') as temporary:
        root=Path(temporary)
        unpack(archive,digest,root)
        summary,complete=read(root/'verified-summary.json'),read(root/'complete.json')
        identity=read(root/'identity.json')
        if (sha(root/'complete.json')!=summary['complete_sha256'] or identity!=complete['identity'] or identity!=summary['identity']
                or identity['probe_commit']!='105f52140f80f562e8d077422fef480931fdd506'
                or identity['optimizer_updates']!=0 or identity['relative_tolerance']!=TOLERANCE):
            raise ValueError('Frozen diagnostic identity differs')
        for name in ('identity.json','cells.jsonl'):
            if sha(root/name)!=complete['artifact_hashes'][name]:raise ValueError('Portable artifact seal differs')
        if sha(root/'prior-first-step-complete.json')!=FIRST_STEP_COMPLETE_SHA or sha(root/'native-training-runtime.json')!=NEW_RUNTIME_SHA:
            raise ValueError('Actual parent velocity or new training condition binding differs')
        prior_complete=read(root/'prior-first-step-complete.json')
        if sha(root/'prior-first-step-cells.jsonl')!=prior_complete['artifact_hashes']['cells.jsonl']:
            raise ValueError('Original full first-step matrix changed')
        runtime=read(root/'native-training-runtime.json')
        prior_rows=jsonl(root/'prior-first-step-cells.jsonl')
        prior={(row['question_id'],row['noise_seed']):row for row in prior_rows}
        rows=jsonl(root/'cells.jsonl')
        if len(rows)!=302 or len(prior)!=302 or summary['cells']!=302 or complete['cells']!=302:
            raise ValueError('Incomplete matrix')
        seen=set()
        passing=0
        peaks={name:{'relative_l2':0.,'relative_max':0.} for name in ('train_vs_native_branch','train_vs_native_cfg','integer_vs_float')}
        expected_artifacts={'identity.json','cells.jsonl'}
        for row in rows:
            key=row['question_id'],row['noise_seed']
            if key in seen or key not in prior or row['artifact']!=prior[key]['artifact']:
                raise ValueError('Changed or repeated diagnostic cell')
            seen.add(key)
            expected_artifacts.add(row['artifact'])
            if (row['condition_sha256']!=runtime['condition_sha256'][key[0]]
                    or not row['native_encoder_and_source_input_bitwise_equal'] or not row['native_first_step_bitwise_reproduced']):
                raise ValueError('GPU condition/input observations differ')
            errors=row['relative_errors']
            if set(errors)!=set(peaks):raise ValueError('Missing comparison arm')
            for name,error in errors.items():
                if set(error)!=set(peaks[name]):raise ValueError('Missing error norm')
                for metric,value in error.items():
                    if not math.isfinite(value) or value<0:raise ValueError('Invalid observed error')
                    peaks[name][metric]=max(peaks[name][metric],value)
            passed=all(value<=TOLERANCE for error in errors.values() for value in error.values())
            if passed!=row['within_registered_tolerance']:raise ValueError('Pass flag contradicts error norms')
            passing+=int(passed)
        if seen!=set(prior) or set(complete['artifact_hashes'])!=expected_artifacts:
            raise ValueError('Incomplete source/velocity coverage')
        if (summary['passing_cells']!=passing or summary['current_and_prior_pts_checked']!=604
                or summary['all_velocity_comparisons_within_tolerance']!=(passing==302)
                or complete['all_velocity_comparisons_within_tolerance']!=(passing==302)):
            raise ValueError('Recount disagrees with full outcome')
        for name,error in peaks.items():
            for metric,value in error.items():
                if not math.isclose(value,summary['max_errors'][name][metric],rel_tol=1e-10,abs_tol=1e-14):
                    raise ValueError('Recorded maximum differs from independent CPU summary')
        return {'archive_sha256':digest,'cells_recounted':302,'passing_cells':passing,'max_errors':peaks,
            'scope':'All portable cells, source/condition identities, pass flags and error maxima recounted locally.604 velocity/input PTs were checked remotely; local artifacts do not contain those tensors. No final-model functional claim.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=verify(args.archive,args.sha256)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(result))
