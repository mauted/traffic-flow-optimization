from simulation import RoadNetwork
import random
import pytest 

random.seed(0)

def test_build_supergraph():
    result = RoadNetwork._build_supergraph(10)
    for i in range(10):
        pre_int = result[i]
        assert len(pre_int.outgoing) > 0