from argparse import Namespace
from pathlib import Path

from scripts.inspire.run_prompt_matching_parallel import build_plan, MODES


def args(tmp_path, phase):
    return Namespace(phase=phase, arm='B', output=tmp_path, base=Path('/models/base'),
                     reader=Path('/models/reader'), prefeval=Path('/upstream/prefeval'),
                     official_source=Path('/upstream/dreamlite'), checkpoint=Path('/parent.pt'),
                     teacher_max_new_tokens=512)


def test_smoke_does_not_launch_writer_or_claim_promotion(tmp_path):
    plan = build_plan(args(tmp_path, 'smoke'))
    jobs = [job for group in plan['groups'] for job in group]
    assert len(jobs) == 4
    assert plan['teacher_steps'] == 12
    assert plan['writer_steps'] is None
    assert plan['scope'] == 'technical_only'
    assert all('writer.py' not in ' '.join(job['command']) for job in jobs)
    assert all('mismatch' not in ' '.join(job['command']) for job in jobs)


def test_pilot_is_balanced_and_does_not_use_heldout_training(tmp_path):
    plan = build_plan(args(tmp_path, 'pilot'))
    jobs = [job for group in plan['groups'] for job in group]
    assert plan['teacher_steps'] == 288 and plan['writer_steps'] == 2048
    assert len(jobs) == 33
    for group in plan['groups']:
        assert len(group) <= 2
        assert len({j['gpu'] for j in group}) == len(group)
    for mode in MODES:
        teachers = [j for j in jobs if j['name'].startswith(f'teacher-{mode}-')]
        assert len(teachers) == 2
        for job in teachers:
            c = job['command']
            assert c[c.index('--split') + 1] == 'pilot'
            assert c[c.index('--steps') + 1] == '288'
        writer = next(j for j in jobs if j['name'] == f'writer-{mode}')['command']
        assert writer[writer.index('--teacher-supervision') + 1] == mode
        assert Path(writer[writer.index('--checkpoint') + 1]) == Path('/parent.pt')
        assert writer[writer.index('--steps') + 1] == '2048'
    teacher_groups = plan['groups'][:3]
    assert [g[0]['command'][g[0]['command'].index('--supervision') + 1] for g in teacher_groups] == list(MODES)
    caches = [j['command'][j['command'].index('--teacher-cache') + 1]
              for j in jobs if '--teacher-cache' in j['command']]
    assert len(caches) == 4 and len(set(caches)) == 1
