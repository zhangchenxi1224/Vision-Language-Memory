"""CPU-only quantitative audit of an immutable exported evidence bundle."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import zipfile
import torch


def rms(z):
    return float(z.double().square().mean().sqrt())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with zipfile.ZipFile(args.evidence) as archive:
        teacher = torch.load(io.BytesIO(archive.read('teacher.pt')), map_location='cpu', weights_only=True)
        student = torch.load(io.BytesIO(archive.read('student.pt')), map_location='cpu', weights_only=True)['latent']
        assert teacher.shape == student.shape == (1, 4, 128, 128)
        report = {'evidence_sha256': hashlib.sha256(args.evidence.read_bytes()).hexdigest(),
                  'shape': list(teacher.shape), 'rms_error': rms(teacher-student), 'curves': {}}
        # Best per-channel affine recalibration is a generous diagnostic for a
        # scale/shift explanation, not a legitimate inference-time correction.
        x, y = student.double().flatten(2), teacher.double().flatten(2)
        xm, ym = x.mean(-1, keepdim=True), y.mean(-1, keepdim=True)
        scale = ((x-xm)*(y-ym)).sum(-1, keepdim=True)/(x-xm).square().sum(-1, keepdim=True)
        shift = ym-scale*xm
        report['best_per_channel_affine'] = {'scale': scale.flatten().tolist(), 'shift': shift.flatten().tolist(),
            'rms_error': rms(scale*x+shift-y), 'remaining_error_fraction': rms(scale*x+shift-y)/rms(x-y)}
        fy, fx = torch.fft.fftfreq(128)[:, None], torch.fft.fftfreq(128)[None, :]
        radius = (fx.square()+fy.square()).sqrt()
        report['spectrum'] = {}
        for name, z in [('teacher', teacher), ('student', student), ('residual', teacher-student)]:
            power = torch.fft.fft2(z.float(), norm='ortho').abs().square()
            report['spectrum'][name] = {'mean': float(z.mean()), 'std': float(z.std()),
                'fractions': {key: float(power[..., mask].sum()/power.sum()) for key, mask in [
                    ('low_lt_0125', radius < .125), ('mid_0125_025', (radius >= .125) & (radius < .25)),
                    ('high_ge_025', radius >= .25)]}}
        for stage in ['single-rank4', 'single-rank16']:
            rows = [json.loads(archive.read(name)) for name in sorted(archive.namelist())
                    if name.startswith(stage+'/metrics/') and name.endswith('.json')]
            curve = {'steps': len(rows), 'clipped_steps': sum(r['gradient_norm_before_clip'] > 1 for r in rows),
                'nonfinite_steps': sum(not all(math.isfinite(r[k]) for k in ['endpoint_mse', 'gradient_norm_before_clip']) for r in rows),
                'max_gradient_norm': max(r['gradient_norm_before_clip'] for r in rows), 'windows': []}
            for lo, hi in [(1, 64), (449, 512), (1537, 1600), (1985, 2048)]:
                selected = [r for r in rows if lo <= r['optimizer_step'] <= hi]
                if selected:
                    curve['windows'].append({'steps': [lo, hi],
                        'mean_mse': statistics.mean(r['endpoint_mse'] for r in selected),
                        'median_gradient_norm': statistics.median(r['gradient_norm_before_clip'] for r in selected)})
            report['curves'][stage] = curve
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
