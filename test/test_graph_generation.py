from road_network_parts import RoadNetwork, RoadSegment, Edge
import random
import pytest 

random.seed(0)

@pytest.fixture(autouse=True)
def reset_state():
    RoadSegment.reset()
    Edge.reset()

def test_build_supergraph():
    result = RoadNetwork._build_supergraph(10)
    for i in range(10):
        pre_int = result[i]
        assert len(pre_int.outgoing) > 0

def test_build_complete_network():
    print("Testing build_complete_network")
    result = RoadNetwork.build_complete_network(10)
    assert result.time == 0