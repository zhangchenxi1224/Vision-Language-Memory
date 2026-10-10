"""Integration contracts for the actual two-host DAG, without launching GPU processes."""
import copy
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.inspire import run_prefeval_aug4_balanced as controller
from scripts.experiments import prefeval_aug4_teacher as teacher
from scripts.experiments import prefeval_aug4_evaluate as evaluate
from scripts.experiments import prefeval_aug4_writer as writer


@pytest.fixture
def args(tmp_path):
    values = {name: tmp_path / name for name in
              ('repo', 'run_root', 'data', 'python', 'base', 'reader', 'official_source', 'prefeval')}
    values.update(host_index=0, host_names=['host0', 'host1'], resume=False)
    result = SimpleNamespace(**values)
    controller.save(result.data, {'train': [{'base_pair_id': f'topic:{i:04}'} for i in range(730)]})
    return result


def decision(args, fallback=False):
    controller.save(args.run_root / 'precheck-decision.json',
                    {'fallback': fallback, 'high_learning_rate': 1.5e-4 if fallback else 5e-4})


def parse_task(task):
    module = { 'prefeval_aug4_teacher.py': teacher, 'prefeval_aug4_writer.py': writer,
               'prefeval_aug4_evaluate.py': evaluate}[Path(task.command[1]).name]
    return module.parser().parse_args(task.command[2:])


@pytest.mark.parametrize('fallback', [False, True])
def test_every_actual_task_command_parses_and_hosts_share_work_equally(args, fallback):
    decision(args, fallback)
    tasks = controller.build_tasks(args)
    assert len({task.name for task in tasks}) == len(tasks)
    formal = [task for task in tasks if task.training]
    assert Counter(task.host for task in formal) == {0: 2, 1: 2}
    assert {(task.host, task.gpu) for task in formal} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    for prefix, expected in [('teacher-', 4), ('conditions-', 4)]:
        selected = [task for task in tasks if task.name.startswith(prefix)
                    and task.name[len(prefix):].isdigit()]
        assert Counter(task.host for task in selected) == {0: expected, 1: expected}
    assert Counter(task.host for task in tasks if task.priority[0] == 6)[0] == Counter(
        task.host for task in tasks if task.priority[0] == 6)[1]
    for task in tasks:
        parsed = parse_task(task)
        assert parsed.data == args.data
        if Path(task.command[1]).name == 'prefeval_aug4_evaluate.py':
            assert parsed.reader == args.reader and parsed.prefeval == args.prefeval
        if task.training:
            arm = task.name.removeprefix('formal-')
            assert parsed.init == ('pretrained' if arm.startswith('P') else 'random')
            assert parsed.steps == 93440 and parsed.snapshot_steps == list(controller.STEPS)
            expected_lr = 5e-5 if arm.endswith('L') else (1.5e-4 if fallback else 5e-4)
            assert parsed.learning_rate == expected_lr
            attempt = int(fallback and arm.endswith('H'))
            assert parsed.expected_init_audit == controller.precheck_output(args, arm, attempt) / 'manifest.json'
            assert args.run_root / 'training-released.json' in task.dependencies
        if task.name.startswith('precheck-0-'):
            assert task.numeric_precheck == task.name.endswith('H')
        if task.name.startswith('precheck-1-'):
            assert task.name.endswith('H') and not task.numeric_precheck
            assert parsed.learning_rate == 1.5e-4


def test_all_checkpoint_jobs_wait_for_ready_canonical_and_complete_donor_generation(args):
    decision(args)
    tasks = controller.build_tasks(args)
    for task in tasks:
        if task.priority[0] != 6:
            continue
        parsed = parse_task(task)
        label = f'{parsed.checkpoint.parent.name}-{parsed.checkpoint.stem.rsplit("-", 1)[1]}-{parsed.split}'
        if parsed.mode == 'generate':
            assert parsed.checkpoint.with_suffix('.ready.json') in task.dependencies
        else:
            # Donors can reside on the other host: the entire generation split is a dependency.
            assert all(controller.done(args, f'generate-{label}-{i}') in task.dependencies
                       for i in range(controller.SHARDS[parsed.split]))
        if parsed.split in ('diagnostics', 'opposites'):
            arm = parsed.checkpoint.parent.name
            step = parsed.checkpoint.stem.rsplit('-', 1)[1]
            assert parsed.canonical_images == args.run_root / 'evaluation' / f'{arm}-{step}-train' / 'images'
            assert all(controller.done(args, f'generate-{arm}-{step}-train-{i}') in task.dependencies
                       for i in range(24))
        if parsed.split == 'official':
            assert parsed.checkpoint.name == 'checkpoint-step-093440.pt'
            if parsed.mode == 'generate':
                assert all(controller.done(args, 'formal-' + a) in task.dependencies for a in controller.ARMS)


@pytest.mark.parametrize('failure_arm', [None, 'P-H', 'R-H'])
def test_precheck_decision_waits_for_all_four_and_uses_only_numeric_failure(args, failure_arm):
    for arm in controller.ARMS[:-1]:
        label = f'precheck-0-{arm}'
        path = (args.run_root / 'tasks' / (label + '.numeric-failure.json') if arm == failure_arm
                else controller.done(args, label))
        # Arbitrarily poor/high MCQ accuracy must not change the decision.
        controller.save(path, {'accuracy': 0 if arm.startswith('P') else 1})
    controller.prepare_precheck_decision(args)
    assert not (args.run_root / 'precheck-decision.json').exists()
    arm = controller.ARMS[-1]
    path = (args.run_root / 'tasks' / (f'precheck-0-{arm}.numeric-failure.json') if arm == failure_arm
            else controller.done(args, f'precheck-0-{arm}'))
    controller.save(path, {'accuracy': 0.01})
    controller.prepare_precheck_decision(args)
    selected = controller.read(args.run_root / 'precheck-decision.json')
    assert selected['fallback'] == (failure_arm is not None)
    assert selected['high_learning_rate'] == (1.5e-4 if failure_arm else 5e-4)


@pytest.mark.parametrize('reason, expected', [
    ('nonfinite_fm_loss', True), ('nonfinite_gradient_norm', True),
    ('nonfinite_optimizer_state', True), ('CUDA out of memory', False),
    ('zero_gradient', False), ('missing_receipt', False), ('missing_inference_checkpoint', False),
])
def test_numeric_failure_does_not_treat_infrastructure_errors_as_nonfinite(args, reason, expected):
    path = args.run_root / 'numeric-failure.json'
    controller.save(path, {'reason': reason})
    assert controller.numeric_failure(path) is expected


def manifests(args, fallback=False):
    decision(args, fallback)
    shared = {'shared_training_identity': {'data': 'same', 'targets': 'same', 'conditions': 'same', 'schedule': 'same'},
              'architecture_sha256': 'same', 'parameter_schema_sha256': 'same', 'unet_parameter_count': 123,
              'frozen_module_hashes': {'vae': 'v', 'text_encoder': 't'}, 'official_unet_sha256': 'same',
              'official_unet_config_sha256': 'same', 'runtime': {'device_name': 'H200'}}
    for arm in controller.ARMS:
        attempt = int(fallback and arm.endswith('H'))
        controller.save(controller.done(args, f'precheck-{attempt}-{arm}'), {'returncode': 0})
        controller.save(controller.precheck_output(args, arm, attempt) / 'manifest.json',
                        {**copy.deepcopy(shared), 'initial_state_sha256': arm[0]})


@pytest.mark.parametrize('fallback', [False, True])
def test_training_release_requires_matching_conditions_and_paired_initialization(args, fallback):
    manifests(args, fallback)
    controller.release_training(args)
    released = controller.read(args.run_root / 'training-released.json')
    assert released['batch'] == 4 and released['steps'] == list(controller.STEPS)
    assert released['initial_states'] == {'P-L': 'P', 'P-H': 'P', 'R-L': 'R', 'R-H': 'R'}


@pytest.mark.parametrize('field', ['shared_training_identity', 'architecture_sha256', 'parameter_schema_sha256',
                                 'unet_parameter_count', 'frozen_module_hashes', 'official_unet_sha256',
                                 'official_unet_config_sha256', 'runtime', 'initial_state_sha256'])
def test_training_release_rejects_changed_inputs_or_unpaired_initial_state(args, field):
    manifests(args)
    path = controller.precheck_output(args, 'P-H', 0) / 'manifest.json'
    changed = controller.read(path)
    changed[field] = 'different'
    controller.save(path, changed)
    with pytest.raises(RuntimeError, match='differ'):
        controller.release_training(args)
    assert not (args.run_root / 'training-released.json').exists()


def test_training_release_rejects_random_weights_equal_to_pretrained(args):
    manifests(args)
    for arm in ('R-L', 'R-H'):
        path = controller.precheck_output(args, arm, 0) / 'manifest.json'
        contents = controller.read(path)
        contents['initial_state_sha256'] = 'P'
        controller.save(path, contents)
    with pytest.raises(RuntimeError, match='unexpectedly match'):
        controller.release_training(args)


@pytest.mark.parametrize('mode', ['train', 'precheck'])
def test_manifest_only_interruption_can_resume_before_first_optimizer_save(args, mode):
    output = args.run_root / mode
    controller.save(output / 'manifest.json', {'already_bound': True})
    task_command = controller.writer_command(args, 'P-L', mode, output, 5e-5)
    parsed = writer.parser().parse_args(task_command[2:])
    assert parsed.resume


def teacher_receipts(args, benefit=True):
    rows = controller.read(args.data)['train']
    for shard in range(24):
        selected = rows[shard::24]
        total = len(selected) * 3 * 4
        controller.save(controller.done(args, f'teacher-read-{shard}'), {'returncode': 0})
        controller.save(args.run_root / 'teacher-evaluation' / f'read-finished-{shard}.json',
                        {'status': 'completed', 'items': total * 4, 'expected_items': total * 4,
                         'binding': {'data_sha256': controller.sha(args.data), 'kind': 'teacher', 'split': 'train',
                                     'shard': shard, 'shards': 24,
                                     'assignment': [r['base_pair_id'] for r in selected]},
                         'counts': {name: {'total': total, 'correct': total if name == 'memory' and benefit else 0,
                                           'parse_failures': 0} for name in ('memory', 'mismatch', 'blank', 'text')}})


def test_teacher_acceptance_covers_all_730_and_does_not_filter_failures(args):
    teacher_receipts(args)
    controller.finalize_teachers(args)
    assert (args.run_root / 'teacher-accepted.json').exists()
    summary = controller.read(args.run_root / 'teacher-summary.json')
    assert summary['preferences'] == 730
    assert summary['selection'] == 'all_train_preferences_retained'
    assert all(v['total'] == 8760 for v in summary['controls'].values())


def test_teacher_acceptance_requires_memory_benefit_and_complete_coverage(args):
    teacher_receipts(args, benefit=False)
    with pytest.raises(RuntimeError, match='memory benefit'):
        controller.finalize_teachers(args)
    assert not (args.run_root / 'teacher-accepted.json').exists()
    teacher_receipts(args)
    path = args.run_root / 'teacher-evaluation/read-finished-0.json'
    broken = controller.read(path)
    broken['binding']['assignment'] = []
    controller.save(path, broken)
    with pytest.raises(RuntimeError, match='exactly once'):
        controller.finalize_teachers(args)
