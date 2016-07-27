import itertools
import numpy as np
import numpy.random as rand
from gameanalysis import AGGFN

def congestion_games(num_players, num_facilities, num_required):
    # Used for both AGGFNA and rsgame representations
    strats = list(itertools.combinations(range(num_facilities),num_required))
    num_strats = len(strats)
    strat_mask = np.zeros([num_strats, num_facilities], dtype=bool)
    inds = np.fromiter(
            itertools.chain.from_iterable((row * num_facilities + f for f in facs) for row, facs
            in enumerate(strats)),
            int, num_strats * num_required)
    strat_mask.ravel()[inds] = True
    # Generate value for congestions
    values = rand.random((num_facilities, 3))
    values[:, 0] *= num_facilities  # constant
    values[:, 1] *= -num_required   # linear
    values[:, 2] *= -1              # quadratic

    # AGGFNA construction
    def make_func(i):
        return lambda x: values[i,0]+values[i,1]*x+values[i,2]*x**2
    S = strat_mask
    action_weights = np.hstack((np.zeros([num_strats, num_strats], float), S))
    function_inputs = S.T
    node_functions = [make_func(i) for i in range(num_facilities)]
    game = AGGFN.Sym_AGG_FNA(num_players,num_strats,action_weights,function_inputs,node_functions)
    
    return game
