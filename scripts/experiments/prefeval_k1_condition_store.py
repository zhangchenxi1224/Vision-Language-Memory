"""Keep exact native conditions on shared disk instead of 30,660 RAM tensors."""
import hashlib
import json
from pathlib import Path


class DiskConditions:
    def __init__(self, root, binding):
        encoded = json.dumps(binding, sort_keys=True).encode()
        self.root = Path(root) / hashlib.sha256(encoded).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True)
        from filelock import FileLock
        with FileLock(str(self.root / 'binding.lock')):
            path = self.root / 'binding.json'
            if path.exists():
                assert json.loads(path.read_text()) == binding
            else:
                path.write_text(json.dumps(binding, indent=2) + '\n')

    def path(self, key):
        return self.root / (hashlib.sha256(json.dumps(key).encode()).hexdigest() + '.pt')

    def ensure(self, key, factory):
        import torch
        from filelock import FileLock
        path = self.path(key)
        with FileLock(str(path.with_suffix('.lock'))):
            if not path.exists():
                condition = factory()
                temp = path.with_suffix('.pt.tmp')
                torch.save({'key': list(key), 'condition': condition}, temp)
                temp.replace(path)

    def __getitem__(self, key):
        import torch
        record = torch.load(self.path(key), map_location='cpu', weights_only=True)
        assert record['key'] == list(key)
        return record['condition']
