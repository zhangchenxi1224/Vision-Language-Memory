"""Verify a downloaded complete six-write CLI replay against its sealed reference."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.reporting.verify_broader_outputs_local import unpack
from scripts.reporting.collect_transition_endpoint import read,sha
from scripts.probes.rgb_package_parity import verify


def collect(archive,digest):
    with tempfile.TemporaryDirectory(prefix='cli-evidence-',dir=ROOT/'.cache') as temporary:
        root=Path(temporary)
        unpack(archive,digest,root)
        recorded=read(root/'parity/parity-result.json')
        actual=verify(root/'parity',root/'inference')
        if actual!=recorded:
            raise ValueError('Full local CLI replay recount differs from recorded parity')
        if sha(root/'package/manifest.json')!=actual['package_manifest_sha256']:
            raise ValueError('Exported package manifest differs from actual CLI inference')
        suite=read(root/'suite-status.json')
        if suite['state']!='completed' or suite['stage']!='all_registered_workloads_finished':
            raise ValueError('Require the full completed suite, including failed functional cases')
        return {'archive_sha256':digest,'bytes':Path(archive).stat().st_size,'parity':actual,
            'suite':suite,'scope':'All36 commands/results,30 raw reads,6 write PNGs and final persisted PNG reverified locally. Exported parameter tensors remain remote. CLI parity is separate from functional correctness.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=collect(args.archive,args.sha256)
    args.output.write_bytes((json.dumps(result,indent=2,sort_keys=True)+'\n').encode('utf-8'))
    print(json.dumps({key:result['parity'][key] for key in ('writes_compared','reads_compared','parity_pass','reference_functional_pass')}))
