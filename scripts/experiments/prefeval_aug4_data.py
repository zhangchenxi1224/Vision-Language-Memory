"""Frozen, source-bound data for the 4x3/512-exposure experiment.

Writer inputs are deliberately constructed by ``writer_exchange`` rather than
serializing these records: questions, options, and labels must not enter Writer.
"""
import argparse
import collections
import copy
import csv
import hashlib
import json
from pathlib import Path

from prefeval_k1_data import ROOT, load_records, official_eval_disclosures, sha

REPORT = ROOT / 'reports/prefeval-aug4x3-init-ablation-exposure512-20261009'
DEFAULT_DATA = REPORT / 'data/runtime_data.json'
ACKNOWLEDGMENT = 'Understood.'
DIAGNOSTIC_SALT = 'aug4-eval96-v1:'
COUNTS = {'train': 730, 'dev': 90, 'official': 180, 'diagnostics': 96, 'opposites': 16}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def normalized(text):
    return ' '.join(text.casefold().split())


def select_diagnostic_ids(train):
    by_topic = collections.defaultdict(list)
    for row in train:
        if not row.get('source_caveats'):
            by_topic[row['topic']].append(row['base_pair_id'])
    return [pid for topic in sorted(by_topic) for pid in sorted(
        by_topic[topic], key=lambda p: hashlib.sha256((DIAGNOSTIC_SALT + p).encode()).hexdigest())[:6]]


def writer_exchange(row, form='W0'):
    """The only two messages authorized as current Writer conditioning."""
    return [{'role': 'user', 'content': row['writer_user_forms'][form]},
            {'role': 'assistant', 'content': ACKNOWLEDGMENT}]


def load_runtime(path=DEFAULT_DATA):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    validate(data)
    return data


def validate(data):
    assert data['schema_version'] == 1
    for split, count in COUNTS.items():
        rows = data[split]
        assert len(rows) == count, (split, len(rows), count)
        assert len({r['base_pair_id'] for r in rows}) == count
        for row in rows:
            assert row['acknowledgment'] == ACKNOWLEDGMENT
            assert row['correct_option_index'] == 0
            assert len(row['options']) == 4 and len(set(row['options'])) == 4
            assert all(isinstance(s, str) and s.strip() for s in row['options'])
            assert all(isinstance(s, str) and s.strip() for f in ('writer_user_forms', 'teacher_question_forms') for s in row[f].values())
            assert [m['role'] for m in writer_exchange(row)] == ['user', 'assistant']
    train = {r['base_pair_id']: r for r in data['train']}
    assert set(train).isdisjoint(r['base_pair_id'] for r in data['dev'])
    assert set(train).isdisjoint(r['base_pair_id'] for r in data['official'])
    for r in data['train']:
        assert set(r['writer_user_forms']) == {'W0', 'W1', 'W2', 'W3'}
        assert set(r['teacher_question_forms']) == {'T1', 'T2', 'T3'}
    assert [r['base_pair_id'] for r in data['diagnostics']] == select_diagnostic_ids(data['train'])
    assert set(collections.Counter(r['topic'] for r in data['diagnostics']).values()) == {6}
    for r in data['diagnostics']:
        original = train[r['base_pair_id']]
        assert r['options'] == original['options'] and not r['source_caveats']
        for field, heldout in [('writer_user_forms', 'Wstar'), ('teacher_question_forms', 'Tstar')]:
            assert set(r[field]) == set(original[field]) | {heldout}
            assert all(r[field][k] == v for k, v in original[field].items())
            assert normalized(r[field][heldout]) not in {normalized(v) for v in original[field].values()}, (r['base_pair_id'], heldout)
    for r in data['opposites']:
        original = train[r['origin_id']]
        index = r['new_correct_option_original_index']
        assert index in (1, 2, 3)
        assert r['options'] == [original['options'][index]] + [v for i, v in enumerate(original['options']) if i != index]
        assert r['teacher_question_forms'] == {'T1': original['teacher_question_forms']['T1']}
    return True


def _base_row(pid, topic, split, user, question, options, correct):
    assert correct == 0, (pid, correct)
    return {'base_pair_id': pid, 'topic': topic, 'split': split,
            'writer_user_forms': {'W0': user}, 'teacher_question_forms': {'T1': question},
            'options': list(options), 'correct_option_index': 0,
            'acknowledgment': ACKNOWLEDGMENT}


def build(augmentation, output=DEFAULT_DATA):
    augmentation, output = Path(augmentation), Path(output)
    manifest_path = augmentation / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for rel, binding in manifest['source']['source_files'].items():
        assert sha(ROOT / Path(rel.replace('\\', '/'))) == binding['sha256'], rel
    for rel, binding in manifest['files'].items():
        assert sha(augmentation / rel) == binding['sha256'], rel
    groups = {r['base_pair_id']: r for r in read_jsonl(augmentation / 'data/semantic_groups.jsonl')}
    labels = json.loads((augmentation / 'source/correct_option_indices.json').read_text(encoding='utf-8'))
    train_source = load_records('train')
    assert set(groups) == {r['base_pair_id'] for r in train_source}
    result = {'schema_version': 1, 'train': [], 'dev': [], 'official': [], 'diagnostics': [], 'opposites': []}
    for split in ('train', 'dev'):
        for old in train_source if split == 'train' else load_records('dev'):
            pid = old['base_pair_id']
            row = _base_row(pid, old['topic'], split, old['history'][0]['content'], old['query']['content'],
                            old['options'], old['benchmark']['evaluation_only']['unshuffled_correct_option_index'])
            if split == 'train':
                group = groups[pid]
                assert labels[pid] == 0
                assert group['writer_user_forms']['W0'] == row['writer_user_forms']['W0']
                assert group['teacher_question_forms']['T1'] == row['teacher_question_forms']['T1']
                row.update({k: copy.deepcopy(group[k]) for k in ('writer_user_forms', 'teacher_question_forms', 'source_caveats', 'review_status')})
            result[split].append(row)
    for old in official_eval_disclosures():
        ev = old['evaluation_only']
        result['official'].append(_base_row(old['base_pair_id'], old['topic'], 'official',
            old['input']['disclosure'][0]['content'], old['input']['query']['content'],
            ev['classification_task_options'], ev['unshuffled_correct_option_index']))
    train = {r['base_pair_id']: r for r in result['train']}
    wordings_path = REPORT / 'data/heldout_wordings.tsv'
    with wordings_path.open(encoding='utf-8', newline='') as handle:
        wordings = list(csv.DictReader(handle, delimiter='\t'))
    assert [r['base_pair_id'] for r in wordings] == select_diagnostic_ids(result['train'])
    for wording in wordings:
        row = copy.deepcopy(train[wording['base_pair_id']])
        row['split'] = 'diagnostics'
        row['writer_user_forms']['Wstar'] = wording['Wstar']
        row['teacher_question_forms']['Tstar'] = wording['Tstar']
        row['heldout_review'] = 'individually_authored_and_semantically_self_reviewed_before_model_results'
        result['diagnostics'].append(row)
    opposites_path = REPORT / 'data/opposite_preferences.json'
    for spec in json.loads(opposites_path.read_text(encoding='utf-8')):
        old = train[spec['origin_id']]
        index = spec['new_correct_option_original_index']
        order = [index] + [i for i in range(4) if i != index]
        row = _base_row(spec['origin_id'] + '::opposite', old['topic'], 'opposites', spec['user'],
                        old['teacher_question_forms']['T1'], [old['options'][i] for i in order], 0)
        row.update({k: spec[k] for k in ('origin_id', 'new_correct_option_original_index', 'reason')})
        row['original_option_order'] = order
        row['opposite_protocol'] = 'counterfactual preference with explicit disambiguation; diagnostic only, not a minimal-negation causal test'
        result['opposites'].append(row)
    review = json.loads((augmentation / 'reports/validation_summary.json').read_text(encoding='utf-8'))
    result['metadata'] = {
        'dataset_id': 'prefeval-aug4x3-512-runtime-v1-20261009',
        'augmentation_dataset_id': manifest['dataset_id'],
        'augmentation_manifest_sha256': sha(manifest_path),
        'augmentation_semantic_groups_sha256': sha(augmentation / 'data/semantic_groups.jsonl'),
        'source_bindings': manifest['source']['source_files'],
        'heldout_wordings_sha256': sha(wordings_path),
        'opposite_preferences_sha256': sha(opposites_path),
        'acknowledgment': ACKNOWLEDGMENT,
        'diagnostic_selection': {'salt': DIAGNOSTIC_SALT, 'rule': 'per-topic smallest SHA256(salt+id), exclude source caveats', 'per_topic': 6},
        'counts': COUNTS,
        'source_caveat_count': sum(bool(r.get('source_caveats')) for r in result['train']),
        'official_overlap_pairs': review['inherited_source_content_overlap_pairs'],
        'official_deduplicated_count': 177,
        'review': {'train': 'previous cross-model review; all source caveats retained',
                  'heldout_wordings': 'individual authoring and model self-review, not independent or human certification',
                  'opposites': 'counterfactuals include added disambiguating preferences because negating the source often makes several distractors valid',
                  'model_results_used_for_selection_or_wording': False},
        'input_contract': {'writer': 'selected user form + fixed Understood.; no questions/options/answers',
                           'teacher': 'train only, T1-T3, same options, hard correct answer',
                           'evaluation': 'actual PNG; diagnostics and opposites never enter teacher/FM optimization'},
    }
    validate(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    receipt = {'status': 'complete_structure_source_identity_and_author_semantics_checked',
               'runtime_data_sha256': sha(output), 'counts': COUNTS,
               'checks': ['source file SHA256 matched augmentation manifest', '730 original W0/T1 exact',
                          'all original options and correct index retained', '96 stratified deterministic heldout pairs',
                          'all Wstar/Tstar differ from train wordings', '16 opposite labels reordered consistently',
                          'train/dev/official ID isolation', 'fixed Understood. for all splits'],
               'limitations': result['metadata']['review']}
    output.with_name('runtime_data.validation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--augmentation', type=Path)
    parser.add_argument('--output', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--validate', type=Path)
    args = parser.parse_args()
    if args.validate:
        load_runtime(args.validate)
        print(json.dumps({'validated': str(args.validate), 'sha256': sha(args.validate)}))
    else:
        if not args.augmentation:
            parser.error('--augmentation is required to build data')
        print(json.dumps(build(args.augmentation, args.output), ensure_ascii=False))
