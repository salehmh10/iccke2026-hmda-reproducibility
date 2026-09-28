import json
from package_checks import ROOT, notebook_errors
if __name__=='__main__':
    errors=[f'{p.relative_to(ROOT)}: {e}' for p in ROOT.rglob('*.ipynb') for e in notebook_errors(json.loads(p.read_text(encoding='utf-8')))]
    print('NOTEBOOK OUTPUTS: '+('FAIL' if errors else 'PASS'))
    for error in errors:
        print(error)
    raise SystemExit(bool(errors))
