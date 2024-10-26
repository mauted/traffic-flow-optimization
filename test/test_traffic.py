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
