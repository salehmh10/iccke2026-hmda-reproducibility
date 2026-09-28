from package_checks import ROOT, rows

def test_saved_v1_contract_and_vocabulary_caveat():
    data=rows(ROOT/'results/classification/feature_v1_dictionary.csv')
    assert len(data)==33
    assert len({r['encoded_name'] for r in data})==33
    assert sum(r['kind']=='numeric' for r in data)==9
    assert sum(r['kind']=='category_indicator' for r in data)==24
    assert all(r['production_vocabulary_verified']=='false' for r in data)
