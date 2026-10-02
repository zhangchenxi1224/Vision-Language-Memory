"""List registered evidence roots; never infer validity from file dates."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', default='current_performance',
                        choices=('current_performance', 'protocol_and_data', 'engineering_provenance', 'all'))
    args = parser.parse_args()
    registry = json.loads((ROOT / 'experiments/registry.json').read_text(encoding='utf-8'))
    rows = [r for r in registry['families'] if args.role == 'all' or r['role'] == args.role]
    for row in rows:
        path = (ROOT / row['path']).resolve()
        if not path.is_relative_to(ROOT) or not path.is_dir():
            raise ValueError(f"Missing or unsafe registered directory: {row['path']}")
    print(json.dumps({'selection_rule': registry['selection_rule'], 'families': rows}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
