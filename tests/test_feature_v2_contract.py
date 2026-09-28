import json
from package_checks import ROOT, rows

def test_selected_features_match_saved_candidate_decisions():
    selected=rows(ROOT/'results/classification/feature_v2_interactions_33.csv')
    candidates=rows(ROOT/'results/classification/interaction_candidates_98.csv')
    assert len(selected)==33
    assert {r['feature'] for r in selected}=={r['feature'] for r in candidates if r['selected']=='True'}
    assert all(r['family']=='onehot_numeric' for r in selected)
    confirmed=json.loads((ROOT/'configs/classification/confirmed_feature_config.json').read_text())
    assert confirmed['eligible_configurations']==[]
    assert confirmed['selected_configuration']=='onehot_numeric'
