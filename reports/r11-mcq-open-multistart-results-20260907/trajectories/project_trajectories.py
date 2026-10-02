"""Read archived tensors only; emit an audited common projection on stdout."""
import hashlib
import io
import json
import pickletools
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path('/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11-open/multistart-5d06b76-20260907-round02')
started = time.time()
dim = 65536
arrays = np.empty((8, 257, dim), dtype=np.float32)
index_hashes = []
tensor_hashes = []
for seed in range(8):
    run = ROOT / f'runs/noise-seed-{seed:02d}-mcq'
    idx_bytes = (run / 'latent_index.jsonl').read_bytes()
    index_hashes.append(hashlib.sha256(idx_bytes).hexdigest())
    rows = [json.loads(line) for line in idx_bytes.splitlines()]
    assert [r['optimizer_step'] for r in rows] == list(range(257))
    for step, row in enumerate(rows):
        raw = (run / row['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row['file_sha256']
        tensor_hashes.append(row['file_sha256'])
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            prefix = z.namelist()[0].split('/')[0]
            assert z.read(prefix + '/byteorder') == b'little'
            meta = z.read(prefix + '/data.pkl')
            assert b'FloatStorage' in meta
            ops = list(pickletools.genops(meta))
            assert any(arg == dim for _, arg, _ in ops)
            stores = [n for n in z.namelist() if '/data/' in n]
            assert stores == [prefix + '/data/0']
            storage = z.read(stores[0])
            assert len(storage) == dim * 4
            arrays[seed, step] = np.frombuffer(storage, dtype='<f4')
    assert np.isfinite(arrays[seed]).all()

# One fixed coordinate system. Exact PCA is fitted to 136 equally spaced
# snapshots (all seeds, steps 0,16,...256); every saved step is projected.
sample = arrays[:, ::16, :].reshape(-1, dim).astype(np.float64)
fit_center = sample.mean(axis=0)
sample -= fit_center
gram = sample @ sample.T
eigenvalues, vectors = np.linalg.eigh(gram)
order = np.argsort(eigenvalues)[::-1]
eigenvalues, vectors = eigenvalues[order], vectors[:, order]
basis = sample.T @ (vectors[:, :3] / np.sqrt(eigenvalues[:3]))
for col in range(3):
    pivot = np.argmax(np.abs(basis[:, col]))
    if basis[pivot, col] < 0:
        basis[:, col] *= -1
assert np.max(np.abs(basis.T @ basis - np.eye(3))) < 1e-10
center = arrays.mean(axis=(0, 1), dtype=np.float64)

paths = []
total_variation = 0.0
projected_variation = np.zeros(3)
for seed in range(8):
    a = arrays[seed].astype(np.float64)
    centred = a - center
    coords = centred @ basis / np.sqrt(dim)
    total_variation += float(np.square(centred).sum()) / dim
    projected_variation += np.square(coords).sum(axis=0)
    from_start = np.sqrt(np.square(a - a[0]).mean(axis=1))
    to_final = np.sqrt(np.square(a - a[-1]).mean(axis=1))
    update = np.r_[0.0, np.sqrt(np.square(np.diff(a, axis=0)).mean(axis=1))]
    travelled = np.cumsum(update)
    pct_path = travelled / travelled[-1]
    mcq = [json.loads(line) for line in (ROOT / f'runs/noise-seed-{seed:02d}-mcq/mcq_endpoint.jsonl').read_text().splitlines()]
    assert len(mcq) == 4 and all(row['correct'] for row in mcq)
    metrics = [json.loads(line) for line in (ROOT / f'runs/noise-seed-{seed:02d}-mcq/metrics.jsonl').read_text().splitlines()]
    paths.append({
        'seed': seed, 'points': np.round(coords, 8).tolist(),
        'fromStart': np.round(from_start, 8).tolist(),
        'toFinal': np.round(to_final, 8).tolist(),
        'update': np.round(update, 10).tolist(),
        'pathLength': float(travelled[-1]),
        'step95Path': int(np.argmax(pct_path >= .95)),
        'endpointRMSEFromStart': float(from_start[-1]),
        'endpointDistanceFrom2DPlane': float(np.sqrt(max(0, np.square(centred[-1]).mean() - np.square(coords[-1, :2]).sum()))),
        'mcqCorrect': 4, 'mcqViews': 4,
    })

initial = arrays[:, 0].astype(np.float64)
final = arrays[:, -1].astype(np.float64)
pairwise = lambda a: np.sqrt(np.maximum(0, np.square(a[:, None] - a[None, :]).mean(axis=-1)))
d0, d1 = pairwise(initial), pairwise(final)
geometry = [json.loads(s) for s in (ROOT / 'geometry_mcq.jsonl').read_text().splitlines()]
assert len(geometry) == 257
max_distance_error = float(np.max(np.abs(d1 - np.asarray(geometry[-1]['raw_latent']['rmse_matrix']))))
assert max_distance_error < 1e-7
for step in [0, 16, 64, 128, 256]:
    for seed in range(8):
        actual = paths[seed]['fromStart'][step]
        expected = geometry[step]['per_seed'][seed]['to_own_start']['rmse']
        assert abs(actual - expected) < 1e-7
endpoint_centered = final - final.mean(axis=0)
endpoint_scores = endpoint_centered @ basis / np.sqrt(dim)
endpoint_total = float(np.square(endpoint_centered).sum()) / dim
endpoint_coverage = np.square(endpoint_scores).sum(axis=0) / endpoint_total
mask = np.triu_indices(8, 1)
result = {
    'schema': 'r11.multistart.projection.v1', 'arm': 'mcq',
    'target': 'ambient', 'shape': [8, 257, dim],
    'sourceRoot': str(ROOT), 'sourceIndexHashes': index_hashes,
    'fileSHA256Verified': len(tensor_hashes),
    'aggregateFileHashesSHA256': hashlib.sha256(''.join(tensor_hashes).encode()).hexdigest(),
    'projection': {
        'method': 'exact_pca_from_common_every_16_step_snapshots',
        'fitSteps': list(range(0, 257, 16)), 'fitCount': 136,
        'fitExplainedVariance': (eigenvalues[:3] / eigenvalues.sum()).tolist(),
        'allStepsCapturedSquaredDistance': (projected_variation / total_variation).tolist(),
        'endpointCapturedVariance': endpoint_coverage.tolist(),
        'coordinateScale': 'PCA scores / sqrt(65536); orthogonal projected RMSE coordinates',
        'centerSHA256': hashlib.sha256(center.tobytes()).hexdigest(),
        'basisSHA256': hashlib.sha256(basis.tobytes()).hexdigest(),
    },
    'paths': paths, 'initialDistances': np.round(d0, 8).tolist(),
    'endpointDistances': np.round(d1, 8).tolist(),
    'meanInitialPairwiseRMSE': float(d0[mask].mean()),
    'meanFinalPairwiseRMSE': float(d1[mask].mean()),
    'maxDistanceErrorVsSavedAudit': max_distance_error,
    'elapsedSeconds': time.time() - started,
}
print('BEGIN_PROJECTION_JSON')
print(json.dumps(result, ensure_ascii=False, separators=(',', ':')))
print('END_PROJECTION_JSON')
