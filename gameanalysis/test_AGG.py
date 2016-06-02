import nash
from AGGFN import Sym_AGG_FNA
import numpy as np

# Building games
game1 = Sym_AGG_FNA(3,
                ['s0','s1','s2'],
                ['p0','p1'],
                {'s0':['s0','p0'], 's1':['s1','s2'], 's2':['s2','p1'],'p0':['s1'],'p1':['s1']},
                {'s0':[1,1],'s1':[1,1],'s2':[1,1]},
                {'p0':['linear',(1,0)],'p1':['linear',(-1,0)]})
assert game1.players['All'] == 3
assert game1.func_table == {'p0': [0,1,2,3], 'p1':[0,-1,-2,-3]}

#game2 = Sym_AGG_FNA.randomAGG(10,3,3,1,5)
game2 = Sym_AGG_FNA.randomAGG(5,3,2)

# Test json input output
def test_json():
    j = game1.to_json()
    game1_from_json = Sym_AGG_FNA.from_json(j)

def test_min_payoff():
    raise NotImplementedError

def test_deviation(agg):
    rs = agg.to_rsgame()
    for i in range(10000):
        mix = np.random.uniform([0]*3,[3]*3,3) 
        mix = mix / sum(mix)
        p_a = agg.deviation_payoffs(mix)
        p_r = rs.deviation_payoffs(mix,as_array=True)
        if not np.allclose(p_a,p_r):
            print(p_a, p_r)
            print(agg.strategies)
            print(agg.neighbors)
            print(agg.func_table)
            print(agg.utilities)
            print(rs)
            raise AssertionError("Deviation payoff data do not math rsgame")

def test_EQ():
    raise NotImplementedError

test_deviation(game2)
