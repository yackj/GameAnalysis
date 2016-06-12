import numpy as np
import numpy.random as rand

from gameanalysis import gamegen
from gameanalysis import reduction
from gameanalysis import rsgame
from gameanalysis import subgame

from test import testutils


@testutils.apply(testutils.game_sizes())
def test_pure_subgame(players, strategies):
    game = rsgame.BaseGame(players, strategies)
    subgames = subgame.pure_subgame_masks(game)
    expectation = game.num_strategies[None].repeat(game.num_roles, 0)
    np.fill_diagonal(expectation, 1)
    expectation = game.role_repeat(expectation.prod(1))
    assert np.all(subgames.sum(0) == expectation)


def test_subgame():
    game = rsgame.BaseGame([3, 4], [3, 2])
    subg = np.asarray([1, 0, 1, 0, 1], bool)
    devs = subgame.deviation_profiles(game, subg)
    assert devs.shape[0] == 7, \
        "didn't generate the right number of deviating profiles"
    adds = subgame.additional_strategy_profiles(game, subg, 1).shape[0]
    assert adds == 6, \
        "didn't generate the right number of additional profiles"
    subg2 = subg.copy()
    subg2[1] = True
    assert (subgame.subgame(game, subg2).num_all_profiles ==
            adds + subgame.subgame(game, subg).num_all_profiles), \
        "additional profiles didn't return the proper amount"

    serial = gamegen.game_serializer(game)
    sub_serial = subgame.subserializer(serial, subg)
    assert (subgame.subgame(game, subg).num_role_strats ==
            sub_serial.num_role_strats)


@testutils.apply(testutils.game_sizes())
def test_maximal_subgames(players, strategies):
    game = gamegen.role_symmetric_game(players, strategies)
    subs = subgame.maximal_subgames(game)
    assert subs.shape[0] == 1, \
        "found more than maximal subgame in a complete game"
    assert subs.all(), \
        "found subgame wasn't the full one"


@testutils.apply(zip([0, 0.1, 0.4, 0.6]))
def test_missing_data_maximal_subgames(prob):
    game = gamegen.role_symmetric_game([3, 4], [3, 2])
    game = gamegen.drop_profiles(game, prob)
    subs = subgame.maximal_subgames(game)
    assert subs.size == 0 or not subs.all()


@testutils.apply(testutils.game_sizes(allow_big=True), repeat=20)
def test_deviation_profile_count(players, strategies):
    game = rsgame.BaseGame(players, strategies)
    sup = (rand.random(game.num_roles) * game.num_strategies).astype(int) + 1
    inds = np.concatenate([rand.choice(s, x) + o for s, x, o
                           in zip(game.num_strategies, sup, game.role_starts)])
    mask = np.zeros(game.num_role_strats, bool)
    mask[inds] = True

    devs = subgame.deviation_profiles(game, mask)
    assert devs.shape[0] == subgame.num_deviation_profiles(game, mask), \
        "num_deviation_profiles didn't return correct number"
    assert np.sum(devs > 0) == subgame.num_deviation_payoffs(game, mask), \
        "num_deviation_profiles didn't return correct number"

    red = reduction.DeviationPreserving(
        game.num_strategies, game.num_players ** 2, game.num_players)
    dpr_devs = red.expand_profiles(subgame.deviation_profiles(
        game, mask)).shape[0]
    num = subgame.num_dpr_deviation_profiles(game, mask)
    assert dpr_devs == num, \
        "num_dpr_deviation_profiles didn't return correct number"


@testutils.apply(testutils.game_sizes(), repeat=20)
def test_subgame_preserves_completeness(players, strategies):
    """Test that subgame function preserves completeness"""
    game = gamegen.role_symmetric_game(players, strategies)
    assert game.is_complete(), "gamegen didn't create complete game"

    mask = game.random_profiles(game.uniform_mixture())[0] > 0

    sub_game = subgame.subgame(game, mask)
    assert sub_game.is_complete(), "subgame didn't preserve game completeness"

    sgame = gamegen.add_noise(game, 1, 3)
    sub_sgame = subgame.subgame(sgame, mask)
    assert sub_sgame.is_complete(), \
        "subgame didn't preserve sample game completeness"


def test_translate():
    prof = np.arange(6) + 1
    mask = np.array([1, 0, 0, 1, 1, 0, 1, 1, 0, 1], bool)
    expected = [1, 0, 0, 2, 3, 0, 4, 5, 0, 6]
    assert np.all(expected == subgame.translate(prof, mask))
