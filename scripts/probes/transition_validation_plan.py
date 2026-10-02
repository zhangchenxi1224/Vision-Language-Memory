"""New held-out event expressions and noise plans for the transition-trained Writer."""
from __future__ import annotations

from vision_memory.training.latent_bank_unet import stable_seed

ENTITY = 'the indigo desk train 001123'
GOLDS = {'ambient': 'ambient', 'jazz': 'jazz', 'clear': 'no active preference'}


def events(state):
    if state == 'clear':
        return (f'Cancel the previously selected music for {ENTITY}. The saved preference should be empty.',
                f'Reset the music setting for {ENTITY} by removing the old selection.')
    if state not in GOLDS:
        raise ValueError('Unknown registered state')
    return (f'Please update {ENTITY}: from now on its music choice is {state}.',
            f'The latest music selection for {ENTITY} is {state}; save this new choice.')


NOOPS = (f'An unrelated inventory check took place for {ENTITY}. Retain its previous music choice.',
         f'{ENTITY.capitalize()} received a routine maintenance notice. Leave its existing music preference intact.')


def plan(seed=20260913):
    noise = [stable_seed(seed, 'official-transition-wording-confirmation-v1', i) for i in range(16)]
    forbidden = {stable_seed(seed, 'training-noise', i) for i in range(14000)}
    for namespace, count in (('heldout-evaluation-noise', 8), ('official-prompt-control-noise', 4),
                             ('official-full-confirmation-noise-v1', 16),
                             ('official-three-state-confirmation-noise-v1', 16),
                             ('official-three-state-confirmation-noise-v2-cfg1', 16),
                             ('official-rgb-memory-chain-noise-v1', 24),
                             ('official-rgb-memory-chain-noise-v2-cfg1', 24),
                             ('official-source-image-parity-noise-v1', 3)):
        forbidden.update(stable_seed(seed, namespace, i) for i in range(count))
    if len(set(noise)) != 16 or forbidden.intersection(noise):
        raise ValueError('Held-out noises must be fresh')
    singles = []
    for state in GOLDS:
        singles.append({'state': state, 'gold': GOLDS[state], 'style': 'original_training_event', 'noise_seeds': noise})
        for index, event in enumerate(events(state)):
            singles.append({'state': state, 'gold': GOLDS[state], 'style': 'new_wording_' + str(index + 1),
                            'event_text': event, 'noise_seeds': noise[:4]})
    chains = []
    orders = [('ambient', 'jazz', 'clear'), ('jazz', 'ambient', 'clear'),
              ('clear', 'ambient', 'jazz'), ('clear', 'jazz', 'ambient')]
    chain_noises = [stable_seed(seed, 'official-transition-wording-chain-v1', i) for i in range(24)]
    if len(set(chain_noises)) != 24 or set(chain_noises).intersection(forbidden.union(noise)):
        raise ValueError('New chain noises must be fresh')
    for sequence, order in enumerate(orders):
        for repetition in range(4):
            steps = []
            for state in order:
                for operation in ('write', 'noop'):
                    index = len(steps)
                    # Two repetitions use original trained events; two use the
                    # two new expressions. All cases still carry real RGB state.
                    wording = repetition - 2
                    step = {'step': index, 'operation': operation, 'expected_state': state,
                            'gold': GOLDS[state], 'noise_seed': chain_noises[repetition * 6 + index],
                            'event_style': 'original_training_event' if wording < 0 else 'new_wording_' + str(wording + 1)}
                    if wording >= 0:
                        step['event_text'] = events(state)[wording] if operation == 'write' else NOOPS[wording]
                    steps.append(step)
            chains.append({'sequence': sequence, 'repetition': repetition, 'steps': steps})
    return {'single_writes': singles, 'rgb_chains': chains,
            'noise_pairing': 'paired across states or sequence orders only; distinct from all prior runs',
            'scope': 'one entity and three states; no unseen-entity or multi-fact claim'}
