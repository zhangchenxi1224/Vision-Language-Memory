"""Validate the prospectively fixed B730 exposure extension, without changing FM."""


def validate_extension(previous, current, *, step, cursor):
    if (step, cursor, previous.get('steps'), current.get('steps')) != (23360, 93440, 23360, 93440):
        raise ValueError('B730 extension must continue step 23360 through step 93440')
    if current.get('snapshot_steps') != [46720, 70080]:
        raise ValueError('B730 intermediate endpoints must be fixed before training')
    if any(current.get(k) != v for k, v in {'arm':'B','stage':'write','split':'train','effective_batch':4}.items()):
        raise ValueError('Only the existing B730 initial-write experiment may be extended')
    bookkeeping = {'steps', 'snapshot_steps', 'implementation_sha256', 'continued_from_sha256'}
    old = {k:v for k,v in previous.items() if k not in bookkeeping}
    new = {k:v for k,v in current.items() if k not in bookkeeping}
    if old != new:
        changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
        raise ValueError(f'Continuation changed scientific inputs: {changed}')

