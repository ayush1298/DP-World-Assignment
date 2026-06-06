import pytest
from src.models import Container, Position
from src.yard_state import YardState
from solution.my_strategy import MyStrategy


@pytest.fixture
def test_layout():
    return {
        "blocks": {
            "B01": {"bays": 5, "rows": 5, "tiers": 5},
            "B02": {"bays": 5, "rows": 5, "tiers": 5},
        }
    }


@pytest.fixture
def empty_yard(test_layout):
    return YardState(test_layout)


def _make_container(cid="C001", vessel="V1", port="PORT_01", weight="MEDIUM", dep_time="2025-01-10T12:00:00"):
    return Container(
        container_id=cid,
        size=20,
        weight_class=weight,
        vessel_id=vessel,
        port_of_discharge=port,
        departure_time=dep_time
    )


def test_erc_empty_stack_is_zero(empty_yard):
    strategy = MyStrategy()
    strategy.initialize(
        {"blocks": {"B01": {"bays": 5, "rows": 5, "tiers": 5}}},
        {"containers": []}
    )
    
    new_lrk = strategy._get_lrk_from_event(_make_container(dep_time="2025-01-15T12:00:00"))
    erc = strategy._compute_erc(empty_yard, "B01", 1, 1, new_lrk)
    assert erc == 0


def test_erc_detects_burial(empty_yard):
    strategy = MyStrategy()
    strategy.initialize(
        {"blocks": {"B01": {"bays": 5, "rows": 5, "tiers": 5}}},
        {"containers": []}
    )
    
    # Place a container with departure time Jan 5 at tier 1
    c1 = _make_container(cid="C1", dep_time="2025-01-05T12:00:00")
    empty_yard.place_container(c1, Position("B01", 1, 1, 1))
    
    # Try to place a container with departure time Jan 10 (later) on top
    # Since new container departs LATER, it will bury C1
    c2 = _make_container(cid="C2", dep_time="2025-01-10T12:00:00")
    new_lrk = strategy._get_lrk_from_event(c2)
    
    erc = strategy._compute_erc(empty_yard, "B01", 1, 1, new_lrk)
    assert erc == 1


def test_erc_zero_compatible_stack(empty_yard):
    strategy = MyStrategy()
    strategy.initialize(
        {"blocks": {"B01": {"bays": 5, "rows": 5, "tiers": 5}}},
        {"containers": []}
    )
    
    # Place a container with departure time Jan 10 at tier 1
    c1 = _make_container(cid="C1", dep_time="2025-01-10T12:00:00")
    empty_yard.place_container(c1, Position("B01", 1, 1, 1))
    
    # Try to place a container with departure time Jan 5 (earlier) on top
    # Since new container departs EARLIER, it does not bury C1 (compatible stack)
    c2 = _make_container(cid="C2", dep_time="2025-01-05T12:00:00")
    new_lrk = strategy._get_lrk_from_event(c2)
    
    erc = strategy._compute_erc(empty_yard, "B01", 1, 1, new_lrk)
    assert erc == 0


def test_fallback_always_returns_valid(empty_yard):
    strategy = MyStrategy()
    strategy.initialize(
        {"blocks": {"B01": {"bays": 2, "rows": 2, "tiers": 5}}},
        {"containers": []}
    )
    
    # Fill up block B01 to height 5
    for bay in (1, 2):
        for row in (1, 2):
            for tier in range(1, 6):
                c = _make_container(cid=f"C_{bay}_{row}_{tier}")
                empty_yard.place_container(c, Position("B01", bay, row, tier))
                
    pos = strategy._fallback_greedy(empty_yard)
    assert pos is not None
