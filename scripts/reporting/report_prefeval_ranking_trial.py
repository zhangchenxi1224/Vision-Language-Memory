"""Summarize verified Plan07 outcomes without changing any scoring rule."""
import json,sys,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'reports/prefeval-rgb-20260917';run=R/'ranking-learning-trial-v1-run'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def fraction(v):return f'{v[0]}/{v[1]} ({v[0]/v[1]:.1%})'
def main():
    g=read(run/'generation-verified.json');v=read(run/'final-verified.json')
    lines=['# Plan07: fixed paired ranking learning trial','',
      'Execution commit: `cb6a6c4`; instance: `dl-clear-retain-h200x4-20260914`.',
      'All 40 archived endpoints received 128 additional updates in each arm with fresh Adam. '
      'A used recovery only; B alternated recovery with full-action listwise ranking. '
      'All 80 endpoints froze before reserved evaluation. Plan05/06 failed gates remain unchanged.', '',
      'The fixed trial completed: 10,240 latent updates, 38,656 gradient forwards, '
      '3,408 strict generations and 2,712 ranking decisions. No Writer updates or final-test calls.', '',
      '## Application outcomes','', '| Panel | Parent | A | B |','|---|---:|---:|---:|']
    for panel in ('application_training','application_reserved','application_training_rotations'):
        lines.append('| Ranking '+panel+' | '+' | '.join(fraction(v['counts'][f'{c}/{panel}/all']) for c in ('parent','A','B'))+' |')
    for panel in ('application_training','application_reserved'):
        lines.append('| Strict generation '+panel+' | '+' | '.join(fraction(g['counts'][f'{c}/{panel}/all']) for c in ('parent','A','B'))+' |')
    lines.append('| Original MCQ | '+fraction(g['parent_counts']['mcq/all'])+' | '+ ' | '.join(fraction(g['counts'][f'{c}/mcq/all']) for c in ('A','B'))+' |')
    lines.extend(['','Reserved ranking semantic-group macros: '+', '.join(f'{c}={v["macro"][c+"/application_reserved"]:.6f}' for c in ('parent','A','B','text','blank'))+'.',
      '', '## Complete recovery','', '| Capacity | A | B |','|---|---:|---:|'])
    for k in ('K1','K2','K3','K4'):
        lines.append('| '+k+' | '+' | '.join(fraction(g['complete'][c]['capacity'][k]) for c in ('A','B'))+' |')
    lines.extend(['','Exact recovery on the training question forms: '+', '.join(f'{c}='+fraction(g['counts'][c+'/recovery_training/all']) for c in ('A','B'))+'.',
      'Exact recovery on the held-out question forms: '+', '.join(f'{c}='+fraction(g['counts'][c+'/qualification_recovery/all']) for c in ('A','B'))+'.',
      'Thus B loses recovery even on trained formulations; the deficit is not only wording generalization.'])
    for c in ('A','B'):
        x=g['complete'][c]
        lines.extend(['',f'{c}: complete slot results `{json.dumps(x["complete_slots"],sort_keys=True)}`; '
            f'selective-clear states {sum(s["selective_clear_complete"] for s in x["selective_clear_states"])}/{len(x["selective_clear_states"])}; '
            f'offline teacher chains {sum(s["complete"] for s in x["offline_teacher_chains"])}/4.'])
    lines.extend(['','These are independently optimized endpoint teachers, not recurrent Writer rollouts.',
      '', '## Overwrite and rotation behavior',''])
    for c in ('parent','A','B','text','blank'):
        pairs=[r for r in v['overwrite_contrasts'] if r['condition']==c and r['panel']=='application_reserved']
        rotation=[r for r in v['rotation_consistency'] if r['condition']==c]
        lines.append(f'- {c}: reserved overwrite both-correct {sum(r["both_correct"] for r in pairs)}/{len(pairs)}; '
            f'training contrast cases correct in all four rotations {sum(r["all_four_correct"] for r in rotation)}/{len(rotation)}.')
    lines.extend(['','## Adoption targets',''])
    lines.extend(f'- {k}: {"PASS" if ok else "FAIL"}' for k,ok in v['adoption_targets'].items())
    lines.extend(['','The application ranking gain is substantial within these authored semantic schemas. '
      'The complete-recovery and original-MCQ targets are still unmet, so this is not an adopted usable memory version. '
      'Strict full-action generation remains a distinct unchanged metric, including failures that add option labels. '
      'Reserved scenarios share authored preference schemas; this result does not establish general long-term-memory transfer.',
      '', '## Compute and evidence','',
      f'Gradient forwards A={v["gradient_forwards"]["A"]}, B={v["gradient_forwards"]["B"]}; '
      f'actual evaluation candidate forwards={v["actual_candidate_forwards"]}. Fixed updates do not imply equal compute.',
      f'Training processed tokens: `{json.dumps(v["training_tokens"],sort_keys=True)}`; '
      f'evaluation processed tokens={v["evaluation_processed_tokens"]}.',
      'Validation: 51 focused tests passed. Both complete Plan07 receipts were reconstructed locally '
      'from raw records byte-identically to the remote receipts. Both historical Plan05/06 failed receipts '
      'also reconstructed byte-identically. No model calls were added by verification.',
      'The archive preserves latents, fresh optimizer states, exact recovery checkpoints, PNGs, '
      'training traces, input-token-bearing generations, candidate scores and full condition-specific prompt proofs.',
      'Archive SHA256: `abbf94cf787fbf835996acd542ee2fc36577fef9b1f4ffa51747f3d8ff5c37d1`.',
      'The archive is distributed as three ordered binary parts; concatenate them before extracting. '
      'See `ranking-learning-trial-archive.json` for part hashes and reconstruction order.', ''])
    (R/'ranking-learning-trial-results.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print('\n'.join(lines))
if __name__=='__main__':main()
