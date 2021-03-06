import pytest

from gameanalysis import agggen


@pytest.mark.parametrize('players,strategies,functions', [
    (2 * [1], 1, 1),
    (2 * [1], 2, 2),
    (2 * [2], 1, 2),
    (2 * [2], 2, 2),
    (5 * [1], 2, 3),
    (2 * [1], 5, 3),
    (2 * [2], 5, 4),
    ([1, 2], 2, 5),
    ([1, 2], [2, 1], 4),
    (2, [1, 2], 4),
    ([3, 4], [2, 3], 6),
])
def test_random_sum_game(players, strategies, functions):
    game = agggen.random_aggfn(players, strategies, functions)
    agggen.serializer(game)


@pytest.mark.parametrize('players,strategies,functions', [
    (2 * [1], 1, 1),
    (2 * [1], 2, 2),
    (2 * [2], 1, 2),
    (2 * [2], 2, 2),
    (5 * [1], 2, 3),
    (2 * [1], 5, 3),
    (2 * [2], 5, 4),
    ([1, 2], 2, 5),
    ([1, 2], [2, 1], 4),
    (2, [1, 2], 4),
    ([3, 4], [2, 3], 6),
])
def test_random_role_game(players, strategies, functions):
    game = agggen.random_aggfn(players, strategies, functions, by_role=True)
    agggen.serializer(game)


@pytest.mark.parametrize('players,facilities,required', [
    (1, 1, 1),
    (2, 2, 1),
    (2, 2, 2),
    (2, 3, 2),
    (3, 4, 2),
    (4, 3, 2),
    (5, 6, 4),
])
def test_congestion_game(players, facilities, required):
    game = agggen.congestion(players, facilities, required)
    serial = agggen.function_serializer(game)
    # Check that function serial has the right format
    assert all(sum(c == '_' for c in strat) == required - 1
               for strat in serial.strat_names[0])


@pytest.mark.parametrize('players,strategies', [
    (1, 1),
    (2, 2),
    (2, 2),
    (2, 3),
    (3, 4),
    (4, 3),
    (5, 6),
])
def test_local_effect_game(players, strategies):
    """Test that deviation payoff formulation is accurate"""
    agggen.local_effect(players, strategies)
