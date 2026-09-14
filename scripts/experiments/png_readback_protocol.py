"""Complete PNG acceptance, fixed before the historical-wording endpoint is observed."""
from scripts.experiments.fresh_wording_validation import PARENT_COMMIT

SOURCES = {
    'registered': ('7b82309eb49e913d8f028230ac5f79c743d28a46', '7b82309-logical'),
    'fresh_wording_v1': ('56b56a8e845821867d599dd795b166800cc9e300', '56b56a8-fresh-wording'),
}
LANES = {'confirmation': ('single_writes', None, 390, 360, 78),
         'chains': ('rgb_chains', None, 480, 480, 96),
         'prefix0': ('historical_prefixes', 0, 560, 480, 112),
         'prefix1': ('historical_prefixes', 1, 560, 480, 112)}


def source_spec(continuation_validation_commit=None):
    if continuation_validation_commit is None:
        return SOURCES, PARENT_COMMIT
    commit = continuation_validation_commit
    if len(commit) != 40 or any(char not in '0123456789abcdef' for char in commit):
        raise ValueError('Require the explicit full continuation validation commit')
    return {'registered': (commit, commit[:7] + '-logical'),
        'fresh_wording_v1': (commit, commit[:7] + '-fresh-wording')}, '4fbc85725d78427235757ace2661d086b896a97f'


def plan(continuation_validation_commit=None):
    sources, parent = source_spec(continuation_validation_commit)
    value = {'policy': 'complete-png-readback-v1', 'parent_commit': parent,
        'sources': {key: {'commit': item[0], 'prefix': item[1]} for key, item in sources.items()},
        'lanes': {key: dict(zip(('mode', 'prefix_lane', 'raw_rows', 'matched_rows', 'images'), value))
                  for key, value in LANES.items()},
        'total_raw_rows': 3980, 'total_matched_rows': 3600, 'total_images': 796,
        'pixel_path': 'PIL RGB PNG -> numpy uint8 copy -> NCHW float32 / 255 -> frozen Reader',
        'generation': {'do_sample': False, 'max_new_tokens': 32, 'gold_before_generation': False},
        'scoring': 'original sealed GOLD_IDS plus immediate EOS 151645; no normalization',
        'controls': 'all blank/donor source PT images rounded to uint8 PNG before reading',
        'chain_parity': 'all 960 original chain pixels and generated token sequences must match',
        'selection': 'all registered cells regardless of prior outcome; no training, retries or best-of selection',
        'scope': 'PNG deployment for seen entities and semantic questions, including fixed fresh event wording'}
    if continuation_validation_commit is not None:
        value['scope'] = 'PNG deployment for seen entities and questions; both complete expression suites are previously observed continuation regressions.'
    return value


def png_pixels(image):
    """The same pixel conversion used by scripts/inference/rgb_memory.py::read_image."""
    import numpy as np
    import torch
    if image.mode != 'RGB' or image.size != (1024, 1024):
        raise ValueError('Require an actual 1024 x 1024 RGB image')
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.


def compare(rows, original, registered, bank, mode, lane):
    from scripts.reporting.collect_broader_validation import summarize_rows
    from scripts.reporting.collect_broader_endpoint import strict_pass
    before, _, _ = summarize_rows(original, registered, bank, mode, lane)
    after, _, _ = summarize_rows(rows, registered, bank, mode, lane)
    key = lambda row: (row['case'], row['noise_seed'], row['prompt_id'])
    old = {key(row): row for row in original}
    counts = {'retained_correct': 0, 'fixed': 0, 'regressed': 0, 'still_wrong': 0}
    chain_equal = 0
    for row in rows:
        previous = old[key(row)]
        if row['source_image_sha256'] != previous['image_sha256']:
            raise ValueError('Readback refers to different source pixels')
        equal = (row['image_sha256'] == previous['image_sha256']
                 and row['generated_token_ids'] == previous['generated_token_ids'])
        if mode == 'rgb_chains':
            chain_equal += int(equal)
        if row['condition'] == 'matched':
            a, b = strict_pass(previous, previous['gold']), strict_pass(row, row['gold'])
            counts[{(True, True): 'retained_correct', (False, True): 'fixed',
                    (True, False): 'regressed', (False, False): 'still_wrong'}[a, b]] += 1
    return {'original': before, 'png': after, 'paired_matched': counts,
            'chain_rows_exact': chain_equal,
            'chain_parity_passed': mode != 'rgb_chains' or chain_equal == 480}
