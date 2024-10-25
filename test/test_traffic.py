from simulation import Traffic
from graph import Road

def test_path_generation():
    nodes = [Road() for _ in range(10)]
    traffic = Traffic(nodes, [], [])
    paths = traffic.generate_start_end(10)
    assert len(paths) == 10
    for path in paths:
        assert path[0] != path[1]
    