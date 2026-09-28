import pytest
from src.common.contracts import routing_strength, synthetic_primary

@pytest.mark.parametrize('probability,expected',[(0,0),(.75,0),(.875,.375),(1,.75)])
def test_probability_dependent_ramp(probability,expected):
    assert routing_strength(probability)==expected

def test_signed_uncapped_residual():
    assert synthetic_primary(100,-400,1)==-200
    assert synthetic_primary(100,40,.875)==115

@pytest.mark.parametrize('value',[-.01,1.01,float('nan'),float('inf')])
def test_invalid_probability(value):
    with pytest.raises(ValueError):
        routing_strength(value)
