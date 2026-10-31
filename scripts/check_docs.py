"""Offline documentation checks, not a certification of translation meaning."""
from pathlib import Path
import re
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = [ROOT/'README.md'] + [ROOT/f'docs/translations/README.{lang}.md' for lang in ['fr','nl','vi']]
    reference = paths[0].read_text(encoding='utf-8')
    commands = re.findall(r'```powershell\n(.*?)```', reference, re.S)
    sections = re.findall(r'<a id="([^"]+)"></a>', reference)
    for path in paths:
        text = path.read_text(encoding='utf-8')
        assert re.findall(r'```powershell\n(.*?)```', text, re.S) == commands, path
        assert re.findall(r'<a id="([^"]+)"></a>', text) == sections, path
        assert len(re.findall(r'^# ', text, re.M)) == 1, path
    for path in ROOT.rglob('*.md'):
        if any(x in {'build','dist','.venv'} or x.endswith('.egg-info') for x in path.relative_to(ROOT).parts):
            continue
        text = path.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if target.startswith(('https://','http://','mailto:')):
                continue  # External availability is not an offline assertion.
            name, _, anchor = target.partition('#')
            destination = (path.parent/name).resolve() if name else path
            assert destination.exists(), f'{path}: missing {target}'
            if anchor and destination.suffix == '.md':
                assert f'id="{anchor}"' in destination.read_text(encoding='utf-8'), f'Missing explicit anchor: {target}'
    for path in (ROOT/'notebooks').glob('*.ipynb'):
        notebook = json.loads(path.read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                assert not cell['outputs'] and cell['execution_count'] is None, path
    print('PASS: README commands, section order, local targets and clean notebook outputs.')


if __name__ == '__main__':
    main()
