"""Run all static package checks; does not change files."""
from package_checks import run

if __name__=='__main__':
    results=[run(name) for name in ['safety','scopes','traceability','links','manifest']]
    raise SystemExit(0 if all(results) else 1)
