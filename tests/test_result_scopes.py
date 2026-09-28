import csv
from package_checks import scopes,traceability

def test_scopes_and_traceability():
    assert scopes()==[]
    assert traceability()==[]

def test_invalid_role_and_duplicate_detected(tmp_path):
    folder=tmp_path/'results'/'classification';folder.mkdir(parents=True)
    row={'source_artifact':'aggregate.csv','source_row_key':'a','source_generation':'historical','source_evaluation_scope':'validation','evaluation_scope':'invented','scientific_status':'authoritative_manuscript_result'}
    with (folder/'synthetic.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerows([row,row])
    errors=scopes(tmp_path)
    assert any('invalid result role' in e for e in errors)
    assert any('duplicate' in e for e in errors)
    assert any('historical generation promoted' in e for e in errors)
