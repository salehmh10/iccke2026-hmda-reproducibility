from package_checks import run
if __name__=='__main__':
    results=[run('safety'),run('manifest')]
    raise SystemExit(0 if all(results) else 1)
