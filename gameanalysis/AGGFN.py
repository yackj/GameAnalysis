import numpy as np
import scipy.special as sps
from scipy.misc import comb
from scipy.special import gammaln
from numpy import random as rand
from itertools import combinations_with_replacement as CwR

from gameanalysis import rsgame
import sys

_TINY = float(np.finfo(np.float64).tiny)

class Sym_AGG_FNA(rsgame.BaseGame):
    """Action Graph Game with Function Nodes.

    Represented games are symmetric. Action node utilities have additive structure.
    Function nodes are contribution-independent. Function nodes have in-edges only
    from action nodes.
    """
    def __init__(self, num_players, num_strategies, action_weights, function_inputs=[],
                 node_functions=[]):
        """
        Parameters
        ----------
            num_players
            num_strategies
            action_weights: floating point matrix with num_strategies rows and
                    (num_strategies + |node_functions|) columns. Each entry specifies the
                    incoming weight in the action graph for the action node (row).
            function_inputs: boolean matrix with |node_functions| rows and num_strategies
                    columns. Each entry specifies whether the action node (column) is an
                    input to the function node (row).
            node_functions: list of activation functions for each function node. These
                    functions must work correctly when applied to a vector of inputs.
        Class Variables
        ---------------
            self.action_weights
            self.function_inputs
            self.configs
            self.log_dev_reps
            self.func_table
        """
        super().__init__(num_players, num_strategies)
        self.action_weights = np.array(action_weights, dtype=float)
        self.function_inputs = np.array(function_inputs, dtype=bool)
        self.configs = np.arange(num_players+1)[:,None]
        self.log_dev_reps = gammaln(num_players) - gammaln(self.configs[1:]) - \
                            gammaln(num_players - self.configs[:-1])
        self.func_table = np.array([f(self.configs[:,0]) for f in
                                    node_functions], dtype=float).T
        self.num_funcs = self.func_table.shape[1]
        self.num_nodes = self.num_funcs + self.num_strategies[0]
        self._min_payoffs = None
        self._max_payoffs = None


    @staticmethod
    def from_json(json_):
        """
        Build a game from the information stored in a dictionary in the json
        format
        """
        raise NotImplementedError("not yet compatible with array implementation.")
        return Sym_AGG_FNA(json_['players'],
                           json_['strategies'],
                           json_['function_nodes'],
                           json_['action_graph'],
                           json_['utilities'],
                           json_['functions'])


    def min_payoffs(self):
        """Returns a lower bound on the payoffs."""
        if self._min_payoffs is None:
            minima = np.zeros([self.num_strategies[0], self.num_nodes])
            minima[:,-self.num_funcs:] = self.func_table.min(0)
            minima[self.action_weights <= 0] = 0

            maxima = np.zeros(minima.shape)
            maxima[:,-self.num_funcs:] = self.func_table.max(0)
            maxima[self.action_weights >= 0] = 0

            self._min_payoffs = ((minima + maxima) *
                                 self.action_weights).sum(1).min(keepdims=True)
            self._min_payoffs.setflags(write=False)
        return self._min_payoffs.view()


    def max_payoffs(self):
        """Returns an upper bound on the payoffs."""
        if self._max_payoffs is None:
            minima = np.zeros([self.num_strategies[0], self.num_nodes])
            minima[:,-self.num_funcs:] = self.func_table.min(0)
            minima[self.action_weights >= 0] = 0

            maxima = np.zeros(minima.shape)
            maxima[:,-self.num_funcs:] = self.func_table.max(0)
            maxima[self.action_weights <= 0] = 0

            self._max_payoffs = ((minima + maxima) *
                                 self.action_weights).sum(1).max(keepdims=True)
            self._max_payoffs.setflags(write=False)
        return self._max_payoffs.view()


    def deviation_payoffs(self, mix, assume_complete=True, jacobian=False):
        # TODO To add jacobian support.
        assert not jacobian, "Sym_AGG_FNA doesn't support jacobian"
        func_node_probs = mix[:,None].repeat(self.num_funcs, 1)
        func_node_probs[np.logical_not(self.function_inputs)] = 0
        func_node_probs = func_node_probs.sum(0)

        act_conf_probs = np.exp(np.log(mix + _TINY) * self.configs[:-1] +
                  np.log(1 - mix + _TINY) * (self.num_players -
                  self.configs[1:]) + self.log_dev_reps)
        func_conf_probs = np.exp(np.log(func_node_probs + _TINY) *
                  self.configs[:-1] + np.log(1 - func_node_probs + _TINY) *
                  (self.num_players - self.configs[1:]))

        EVs = np.empty(self.num_strategies)
        for s in range(self.num_strategies[0]):
            action_outputs = self.configs[:-1].repeat(self.num_strategies, 1)
            action_outputs[:,s] += 1
            function_outputs = np.array(self.func_table[:-1])
            function_outputs[:,self.function_inputs[s]] = \
                            self.func_table[1:,self.function_inputs[s]]
            node_EVs = np.append((act_conf_probs * action_outputs).sum(0),
                                 (func_conf_probs * function_outputs).sum(0))
            EVs[s] = np.dot(node_EVs, self.action_weights[s])
        return EVs


    def to_json(self):
        """
        Creates a json format of the game for storage
        """
        raise NotImplementedError("not yet compatible with array implementation.")
        json = {}
        json['players'] = self.players['All']
        json['strategies'] = list(self.strategies['All'])
        json['function_nodes'] = self.function_nodes
        action_graph = {s:list(self.neighbors[s].keys()) \
                for s in self.neighbors}
        json['action_graph'] = action_graph
        json['utilities'] = self.utilities
        json['functions'] = self.functions
        return json


    @staticmethod
    def randomAGG(num_players, num_strats, num_FNs, D_min=0, D_max=-1,
                  w_mean=0, w_var=3):
        """
        D_min: minimum in-degree for an action node
        D_max: maximum in-degree for an action node
        w_mean: mean of weights
        w_var: variance of weights
        """
        raise NotImplementedError("not yet compatible with array implementation.")
        if D_min < 0 or D_min >= num_strats:
            D_min = 0
        if D_max < 0 or D_max >= num_strats:
            D_max = num_strats / 2

        # This maps a function to the (mean, var) tuple for params
        func = {
                'quadratic': ([0,0,0],[tiny,2,1]),
                'linear': ([0,0],[2,2])
        }

        strategies = ["s"+str(i) for i in range(num_strats)]
        FNs = ["p"+str(i) for i in range(num_FNs)]
        nodes = strategies + FNs
        action_graph = {}
        utilities = {}
        functions = {}

        # Connect the function nodes first
        for fn in FNs:
            num_neighbors = np.random.randint(num_strats/2, num_strats)
            neighbors = sorted(sample(strategies, num_neighbors))
            action_graph[fn] = neighbors
            func_type = sample(func.keys(),1)[0]
            param = [np.random.normal(m,v) for m,v in zip(*func[func_type])]
            functions[fn] = [func_type, tuple(param)]

        for s, strat in enumerate(strategies):
            num_neighbors = np.random.randint(D_min,D_max+1)
            neighbors = sorted(sample(strategies[:s] + strategies[s+1:] + FNs,\
                                      num_neighbors) + [strat])
            action_graph[strat] = neighbors
            u = [np.random.normal(w_mean,w_var) for neighbor in neighbors]
            utilities[strat] = np.array(u)

        return Sym_AGG_FNA(num_players, strategies, FNs, action_graph,
                           utilities, functions)


    def get_payoffs(self, profile, default=None):
        """
        Returns the payoffs to the given pure strategy profile
        Input:
            profile: numpy array representation of the profile
            default: not used in AGGFN
        Output:
            The payoff array
        """
        assert self.verify_profile(profile)
        func_counts = (self.function_inputs * profile).sum(1)
        func_vals = np.array([self.func_table[n,i] for i, n in enumerate(func_counts)])
        counts = np.append(profile, func_vals)
        payoffs = (self.action_weights * counts).sum(1)
        payoffs[profile == 0] = 0
        return payoffs


    def get_noisy_payoffs(self, profile, mean, var):
        """
        Returns payoffs of a given profile with gaussian noises
        Input:
            profile: numpy array representation of the profile
            mean: mean of the gaussian noise
            var: variance of the gaussian noise
        Return:
            The noisy payoff array
        """
        orig_pay = self.get_payoffs(profile)
        payoffs =  orig_pay + ( var * rand.randn(*orig_pay.shape) + mean )
        payoffs[profile == 0] = 0
        return payoffs


    def to_rsgame(self, prop=1, noise=None, samples=1):
        """
        This method builds an rsgame object that represent the same
        game. 
        Input:
            prop:   the proportion of the profiles that will be used in 
                    constructing the rs_game
            noise:  the parameters for the gaussian noises, tuple of 
                    (mean, var). Will not add noise if noise is None
            sample: number of samples drawn before taking the average
        Return:
            rs_game representation of the game
        """
        all_profs = rsgame.BaseGame(self.num_players,self.num_strategies).all_profiles()
        profiles = all_profs[rand.choice(all_profs.shape[0], \
                                         int(prop * len(all_profs)), \
                                         replace=False)]
        if noise is None:
            payoffs = np.array([self.get_payoffs(profile) for profile in profiles])
        else:
            payoffs = np.empty([samples] + list(profiles.shape))
            for i in range(samples):
                payoffs[i] = np.array([self.get_noisy_payoffs(profile, *noise) \
                                       for profile in profiles])
            payoffs = payoffs.mean(0)
        return rsgame.Game(self.num_players, self.num_strategies, profiles, payoffs)
