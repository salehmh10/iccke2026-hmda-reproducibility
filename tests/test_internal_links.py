from package_checks import link_errors,links

def test_repository_links():
    assert links()==[]

def test_valid_and_missing_links(tmp_path):
    p=tmp_path/'README.md';p.write_text('[good](target.md) [bad](absent.md)')
    (tmp_path/'target.md').write_text('synthetic')
    errors=link_errors(p,tmp_path)
    assert len(errors)==1 and 'absent.md' in errors[0]
