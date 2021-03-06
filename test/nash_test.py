import itertools
import json
import math

import numpy as np
import pytest

from gameanalysis import agggen
from gameanalysis import gamegen
from gameanalysis import gameio
from gameanalysis import nash
from gameanalysis import regret
from gameanalysis import rsgame
from test import testutils


METHODS = [('optimize', {}), ('replicator', {}), ('replicatorode', {}),
           ('fixedpoint', {})]
ALL_METHODS = list(map(dict, itertools.chain.from_iterable(
    itertools.combinations(METHODS, i)
    for i in range(1, len(METHODS) + 1))))


@pytest.mark.parametrize('_', range(20))
def test_pure_prisoners_dilemma(_):
    game = gamegen.prisoners_dilemma()
    eqa = nash.pure_nash(game)

    assert eqa.shape[0] == 1, "didn't find exactly one equilibria in pd"
    expected = [0, 2]
    assert np.all(expected == eqa), \
        "didn't find pd equilibrium"


@pytest.mark.parametrize('methods', ALL_METHODS * 20)
def test_mixed_prisoners_dilemma(methods):
    game = gamegen.prisoners_dilemma()
    eqa = nash.mixed_nash(game, dist_thresh=1e-3, processes=1, **methods)

    assert eqa.shape[0] >= 1, \
        "didn't find at least one equilibria in pd {}".format(eqa)
    assert all(regret.mixture_regret(game, eqm) < 1e-3 for eqm in eqa), \
        "returned equilibria with high regret"
    expected = [0, 1]
    assert np.isclose(eqa, expected, atol=1e-3, rtol=1e-3).all(1).any(), \
        "didn't find pd equilibrium {}".format(eqa)


@pytest.mark.parametrize('methods', ALL_METHODS)
@pytest.mark.parametrize('eq_prob', [0, .1, .2, .3, .5, .7, .8, .9, 1])
def test_mixed_known_eq(methods, eq_prob):
    game = gamegen.sym_2p2s_known_eq(eq_prob)
    eqa = nash.mixed_nash(game, processes=1, **methods)
    assert eqa.shape[0] >= 1, "didn't find equilibrium"
    expected = [eq_prob, 1 - eq_prob]
    assert np.isclose(eqa, expected, atol=1e-2, rtol=1e-2).all(1).any(), \
        "didn't find correct equilibrium {} instead of {}".format(
            eqa, expected)


@pytest.mark.parametrize('eq_prob', (p / 10 for p in range(11)))
def test_optimization_stable_point(eq_prob):
    game = gamegen.sym_2p2s_known_eq(eq_prob)
    opt = nash.RegretOptimizer(game)
    val, grad = opt.grad(np.array([eq_prob, 1 - eq_prob]), 1)
    assert np.isclose(val, 0), \
        "value at equilibrium was not close to zero: {}".format(val)
    assert np.allclose(grad, 0), \
        "grad at equilibrium was not close to zero: {}".format(grad)


def test_pure_roshambo():
    game = gamegen.rock_paper_scissors()
    eqa = nash.pure_nash(game)
    assert eqa.size == 0, "found a pure equilibrium in roshambo"
    eqa = nash.pure_nash(game, 1)
    assert eqa.shape[0] == 3, \
        "didn't find low regret ties in roshambo"
    eqa = nash.pure_nash(game, 2)
    assert eqa.shape[0] == game.num_all_profiles, \
        "found profiles with more than 2 regret in roshambo"


def test_minreg_roshambo():
    game = gamegen.rock_paper_scissors()
    eqm = nash.min_regret_profile(game)
    assert np.all(np.sort(eqm) == [0, 0, 2]), \
        "min regret profile was not rr, pp, or ss"


def test_minreg_grid_roshambo():
    game = gamegen.rock_paper_scissors()
    eqm = nash.min_regret_grid_mixture(game, 3)  # Not enough for eq
    assert np.isclose(regret.mixture_regret(game, eqm), .5), \
        "min regret grid didn't find [.5, .5, 0] profile with regret .5"
    eqm = nash.min_regret_grid_mixture(game, 4)  # hit eqa perfectly
    assert np.isclose(regret.mixture_regret(game, eqm), 0), \
        "min regret grid didn't find equilibrium"


def test_minreg_rand_roshambo():
    game = gamegen.rock_paper_scissors()
    eqm = nash.min_regret_rand_mixture(game, 20)
    assert regret.mixture_regret(game, eqm) < 2 + 1e-7, \
        "Found a mixture with greater than maximum regret"


@pytest.mark.parametrize('methods', ALL_METHODS)
def test_mixed_roshambo(methods):
    game = gamegen.rock_paper_scissors()
    eqa = nash.mixed_nash(game, dist_thresh=1e-2, processes=1, **methods)
    assert eqa.shape[0] == 1, \
        "didn't find right number of equilibria in roshambo"
    assert np.allclose(1 / 3, eqa, rtol=1e-3, atol=1e-3), \
        "roshambo equilibria wasn't uniform"


def test_hard_roshambo():
    game = gamegen.rock_paper_scissors(loss=[-2, -3, -3])
    eqa = nash.mixed_nash(game)
    assert eqa.shape[0] == 1, \
        "didn't find right number of equilibria in roshambo"
    assert np.allclose([0.3125, 0.40625, 0.28125], eqa), \
        "roshambo equilibria wasn't uniform"


def test_at_least_one():
    # Equilibrium of game is not at a starting point for equilibria finding
    game = gamegen.sym_2p2s_known_eq(1 / math.sqrt(2))
    # Don't converge
    opts = {'max_iters': 0}
    eqa = nash.mixed_nash(game, processes=1, replicator=opts)
    assert eqa.size == 0, "found an equilibrium normally"
    eqa = nash.mixed_nash(game, replicator=opts, processes=1,
                          at_least_one=True)
    assert eqa.shape[0] == 1, "at_least_one didn't return anything"


@pytest.mark.parametrize('methods,strategies', zip(
    ALL_METHODS,
    [
        [1],
        [2],
        [1, 1],
        [2, 2],
        [4, 4, 4],
        [1, 3],
    ]))
def test_mixed_nash(methods, strategies):
    game = gamegen.role_symmetric_game(1, strategies)
    eqa = nash.mixed_nash(game, at_least_one=True, processes=1, **methods)
    assert eqa.size > 0, "Didn't find an equilibria with at_least_one on"


def test_empty_game():
    game = rsgame.game(2, 3)
    with pytest.raises(ValueError):
        nash.min_regret_profile(game)


def test_hard_nash():
    with open('test/hard_nash_game_1.json') as f:
        game, serial = gameio.read_game(json.load(f))
    eqa = nash.mixed_nash(game)
    expected = serial.from_mix_json({
        'background': {
            'markov:rmin_30000_rmax_30000_thresh_0.001_priceVarEst_1e6':
            0.5407460907477768,
            'markov:rmin_500_rmax_1000_thresh_0.8_priceVarEst_1e9':
            0.45925390925222315
        },
        'hft': {
            'trend:trendLength_5_profitDemanded_50_expiration_50': 1.0
        }
    })
    assert np.isclose(game.trim_mixture_support(eqa), expected,
                      atol=1e-4, rtol=1e-4).all(1).any(), \
        "Didn't find equilibrium in known hard instance"


@testutils.run_if_big
@pytest.mark.parametrize('_', range(20))
def test_fixed_point_always_eq(_):
    num_roles = np.random.randint(1, 4)
    players = np.random.randint(2, 5, num_roles)
    strategies = np.random.randint(2, 5, num_roles)
    functions = np.random.randint(2, 8)
    agame = agggen.random_aggfn(players, strategies, functions)
    eqa = nash.mixed_nash(agame, fixedpoint=None)
    assert eqa.size, "didn't find equilibrium but should always find one"
