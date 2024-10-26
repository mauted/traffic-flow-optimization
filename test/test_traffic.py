from simulation import Traffic
from graph import Road
import pytest 

@pytest.fixture(autouse=True)
def reset_state():
    Road.reset()

def test_path_generation():
    nodes = [Road() for _ in range(10)]
    traffic = Traffic(nodes, [], [])
    paths = traffic.generate_start_end(10)
    assert len(paths) == 10
    for path in paths:
        assert path[0] != path[1]

def test_traceback():
    nodes = [Road() for _ in range(12)]
    A, B, C, D, E, F, I, K, M, S, W, Z = nodes
    traffic = Traffic(nodes, [], [])
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
    
    nodes = [Road() for _ in range(10)]
    A, B, C, D, E, F, G, H, I, J = nodes
    edges = [
    (A, B),
    (A, C),
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
    (A, D)
    ]
    for source, target in edges:
        source.add_outgoing(target)
        target.add_incoming(source)
        
    traffic = Traffic(nodes, edges, [])
    print("AAAAAAAAAAA")
    print(traffic.dfs(A, B))
    print(traffic.dfs(A, G))
    assert False 
    
    
