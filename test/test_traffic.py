from simulation import RoadNetwork
from graph import RoadSegment
import random
import pytest 

@pytest.fixture(autouse=True)
def reset_state():
    RoadSegment.reset()
    
def numbers_to_letters(nodes):
    
    return [chr(65 + node.id) for node in nodes]

def test_path_generation():
    
    nodes = [RoadSegment() for _ in range(10)]
    traffic = RoadNetwork(nodes, [], [])
    paths = traffic.generate_vehicle_endpoints(10)
    assert len(paths) == 10
    for path in paths:
        assert path[0] != path[1]

def test_traceback():
    
    nodes = [RoadSegment() for _ in range(12)]
    A, B, C, D, E, F, I, K, M, S, W, Z = nodes
    traffic = RoadNetwork(nodes, [], [])
    parents = {
        B: A,
        C: A,
        D: B,
        E: B,
        F: C,
        I: D,
        K: E,
        M: F,
        S: I,
        W: K,
        Z: M
    }
    assert traffic._traceback(W, parents) == [A, B, E, K, W]
    assert traffic._traceback(F, parents) == [A, C, F]
    assert traffic._traceback(Z, parents) == [A, C, F, M, Z]
    assert traffic._traceback(S, parents) == [A, B, D, I, S]
    
def test_dfs(): 
    
    nodes = [RoadSegment() for _ in range(10)]
    A, B, C, D, E, F, G, H, I, J = nodes
    edges = [
    (A, B),
    (A, C),
    (A, D),
    (B, D),
    (B, E),
    (C, F),
    (D, G),
    (E, H),
    (F, I),
    (G, J),
    (H, A),
    (I, B),
    (J, C),
    ]
    for s, t in edges:
        s.add_outgoing(t)
        t.add_incoming(s)

    traffic = RoadNetwork(nodes, edges, [])
    
    # test that the start and end are in the nodes, and that every edge in the path is in the list of edges
    # this guarantees that we have a feasible path through the traffic network 
    for _ in range(10): 
        start, end = random.choices(nodes, k=2)
        route = traffic.dfs(start, end)
        assert route[0] in nodes
        assert route[-1] in nodes
        for i in range(1, len(route)):
            assert (route[i - 1], route[i]) in edges
    