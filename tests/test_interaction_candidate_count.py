from collections import Counter
from package_checks import ROOT, rows

def test_full_candidate_space():
    data=rows(ROOT/'results/classification/interaction_candidates_98.csv')
    assert len(data)==len({r['feature'] for r in data})==98
    assert Counter(r['family'] for r in data)=={'numeric_relative':10,'nonlinear':15,'category_context':40,'onehot_numeric':33}
    assert all(r['formula'] and r['source_features'] and r['inference_fallback'] for r in data)
