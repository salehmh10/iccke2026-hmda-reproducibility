import pytest
from src.common.contracts import denial_orientation, synthetic_average_precision

def test_denial_inverts_labels_and_probabilities():
    labels,scores=denial_orientation([1,0,1,1],[.9,.8,.7,.6])
    assert labels==[0,1,0,0]
    assert scores==pytest.approx([.1,.2,.3,.4])
    assert synthetic_average_precision(labels,scores)==pytest.approx(1/3)
    assert synthetic_average_precision([1,0,1,1],[.9,.8,.7,.6])==pytest.approx(29/36)

def test_ap_ties_use_recall_steps():
    assert synthetic_average_precision([1,0],[.5,.5])==.5

def test_invalid_orientation_rejected():
    with pytest.raises(ValueError):
        denial_orientation([2],[.4])
