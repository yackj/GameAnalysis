import sys
import numpy as np

from gameanalysis import nash
from gameanalysis.AGGFN import Sym_AGG_FNA
from test import testutils

def test_constructor():
    game1 = Sym_AGG_FNA(3,
                    ['s0','s1','s2'],
                    ['p0','p1'],
                    {'s0':['s0','p0'], 's1':['s1','s2'], 's2':['s2','p1'], \
                            'p0':['s1'],'p1':['s1']},
                    {'s0':[1,1],'s1':[1,1],'s2':[1,1]},
                    {'p0':['linear',(1,0)],'p1':['linear',(-1,0)]})
    assert game1.func_table == {'p0': [0,1,2,3], 'p1':[0,-1,-2,-3]}, \
            "Does not produce the desired func_table"


@testutils.apply(((5,5,3),(10,5,3),(50,2,2),(2,50,2)),repeat=3)
def test_min_payoff(players, strats, fns):
    agg = Sym_AGG_FNA.randomAGG(players, strats, fns)
    rs = agg.to_rsgame()
    assert agg.min_payoffs(True) <= rs.min_payoffs(True), \
			"Min payoff does not match that of rsgame"


@testutils.apply(((5,5,3),(10,5,5)),repeat=5)
def test_deviation(players, strats, fns):
    agg = Sym_AGG_FNA.randomAGG(players,strats,fns,1,2)
    rs = agg.to_rsgame()
    mixes = agg.random_mixtures(50,as_array=True)
    for mix in mixes:
        p_a = agg.deviation_payoffs(mix)
        p_r = rs.deviation_payoffs(mix,as_array=True)
        assert np.allclose(p_a,p_r), \
                "Deviation payoff data do not match rsgame"


@testutils.apply(((5,5,3),(10,5,5)),repeat=5)
def test_json(players, strats, fns):
    agg = Sym_AGG_FNA.randomAGG(players, strats, fns)
    j = agg.to_json()
    json_game = Sym_AGG_FNA.from_json(j)
    mixes = agg.random_mixtures(50,as_array=True)
    for mix in mixes:
        p_j = json_game.deviation_payoffs(mix, as_array=True)
        p_a = agg.deviation_payoffs(mix,as_array=True)
        assert np.allclose(p_a,p_j), \
                "Deviation payoff data do not match rsgame"


@testutils.apply(((5,5,3),(10,5,5)),repeat=5)
def test_EQ(players, strats, fns):
    agg = Sym_AGG_FNA.randomAGG(players, strats, fns)
    rep = nash.mixed_nash(agg, 1e-2, 1e-2, 1, None, False, True,
                    replicator={})
    for eq in rep:
        # Make sure that the regret for this profile is less than thres
        eq = agg.as_mixture(eq, as_array=True)
        dev_pay = agg.deviation_payoffs(eq)
        ex_pay = np.dot(dev_pay, eq)
        regret = np.absolute(dev_pay - ex_pay)
        assert np.amax(regret) > 1e-2, \
                "The equilibrium has high regret"

