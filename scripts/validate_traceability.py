from package_checks import run
if __name__=='__main__':
    raise SystemExit(0 if run('traceability') else 1)
