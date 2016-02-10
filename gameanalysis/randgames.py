import random
import itertools
from os import path
import collections

import numpy as np
import numpy.random as rand
import scipy.misc as scm

from gameanalysis import rsgame
from gameanalysis import utils


# Populate word list for generating better names
_WORD_LIST_FILE = path.join(path.dirname(path.dirname(__file__)),
                            '.wordlist.txt')
_WORD_LIST = []

try:
    with open(_WORD_LIST_FILE) as f:
        for word in f:
            _WORD_LIST.append(word[:-1])
except OSError:  # Something bad happened
    pass

default_distribution = lambda shape=None: rand.uniform(-1, 1, shape)


def _as_list(item, length):
    """Makes sure item is a length list"""
    try:
        item = list(item)
        assert len(item) == length, "List was not the proper length"
    except TypeError:
        item = [item] * length
    return item


def _index(iterable):
    """Returns a dictionary mapping elements to their index in the iterable"""
    return dict(map(reversed, enumerate(iterable)))


def _random_strings(number, prefix='x', padding=None, cool=False):
    """Generate random strings without repetition

    If cool (the default) these strings are generated from a word list,
    otherwise they are generated by adding a prefix to a bunch of zero padded
    integers. The padding defaults so that all of the strings are the same
    length.

    """
    if cool and _WORD_LIST:
        return random.sample(_WORD_LIST, number)
    else:
        if padding is None:
            padding = len(str(number - 1))
        return ('{}{:0{}d}'.format(prefix, i, padding) for i in range(number))


def _compact_payoffs(game):
    """Given a game returns a compact representation of the payoffs

    In this case compact means that they're in one ndarray. This representation
    is inefficient for almost everything but an independent game with full
    data.

    Parameters
    ----------
    game : rsgame.Game
        The game to generate a compact payoff matrix for

    Returns
    -------
    payoffs : ndarray; shape (s1, s2, ..., sn, n)
        payoffs[s1, s2, ..., sn, j] is the payoff to player j when player 1
        plays s1, player 2 plays s2, etc. n is the total number of players.

    strategies : [(role, [strat])]
        The first list indexes the players, and the second indexes the
        strategies for that player.
    """
    strategies = list(itertools.chain.from_iterable(
        itertools.repeat((role, list(strats)), game.players[role])
        for role, strats in game.strategies.items()))

    payoffs = np.empty([len(s) for _, s in strategies] + [len(strategies)])
    for profile, payoff in game.payoffs(as_array=True):
        # This generator expression takes a role symmetric profile with payoffs
        # and generates tuples of strategy indexes and payoffs for every player
        # when that player plays the given strategy.

        # The first line takes results in the form:
        # (((r1i1, r1p1), (r1i2, r1p2)), ((r1i1, r2p1),)) that is grouped by
        # role, then by player in the role, then grouped strategy index and
        # payoff, and turns it into a single tuple of indices and payoffs.
        perms = (zip(*itertools.chain.from_iterable(sp))
                 # This product is over roles
                 for sp in itertools.product(*[
                     # This computes all of the ordered permutations of
                     # strategies in a given role, e.g. if two players play s1
                     # and one plays s2, this iterates over all possible ways
                     # that could be expressed in an asymmetric game.
                     utils.ordered_permutations(itertools.chain.from_iterable(
                         # This iterates over the strategy counts, and
                         # duplicates strategy indices and payoffs based on the
                         # strategy counts.
                         itertools.repeat((i, v), c) for i, (c, v)
                         in enumerate(zip(p, pay))))
                     for p, pay in zip(profile, payoff)]))
        for indices, utilities in perms:
            payoffs[indices] = utilities
    return payoffs, strategies


def _symmetric_counts(num_players, num_strategies):
    """Returns an array of the counts for a symmetric game"""
    num_profs = scm.comb(num_strategies, num_players, exact=True,
                         repetition=True)
    profs = (collections.Counter(x) for x
             in itertools.combinations_with_replacement(
                 range(num_strategies), num_players))
    return np.fromiter(
        itertools.chain.from_iterable(
            (prof[i] for i in range(num_strategies)) for prof in profs),
        int, num_profs * num_strategies).reshape((num_profs, num_strategies))


def _gen_matrix_game(matrix, cool=False):
    """Create an asymmetric game from a matrix

    Parameters
    ----------
    matrix : ndarray (nstrats1, ..., nstratsk, k)
        An ndarray representing the payoffs to every player for every
        combination of strategies. A bimatrix game would have shape (strategies
        for player 1, strategies for player 2, 2).
    cool : bool
        Whether to generate word-like names. (default: False)
    """
    num_players = len(matrix.shape) - 1
    assert all(s > 0 for s in matrix.shape), \
        "Matrix was not in the proper shape"
    assert num_players == matrix.shape[-1], \
        "Matrix was not in the proper shape"
    roles = list(_random_strings(num_players, prefix='r', cool=cool))
    strategies = collections.OrderedDict(
        (role, list(_random_strings(num_strat, prefix='s', cool=cool)))
        for role, num_strat
        in zip(roles, matrix.shape))
    players = {role: 1 for role in roles}
    role_strats = [[(role, strat) for strat in strats]
                   for role, strats in strategies.items()]

    profiles = ({role: [(strat, 1, float(payoff))] for (role, strat), payoff
                 in zip(assignment, payoffs)}
                for assignment, payoffs
                in zip(itertools.product(*role_strats),
                       matrix.reshape([-1, num_players])))
    return rsgame.Game(players, strategies, profiles,
                       utils.prod(matrix.shape[:-1]))


def _gen_empty_rs_game(num_roles, num_players, num_strategies, cool=False):
    """Create a role symmetric game"""
    assert num_roles > 0, "Number of roles must be greater than 0"
    num_players = _as_list(num_players, num_roles)
    num_strategies = _as_list(num_strategies, num_roles)

    assert all(p > 0 for p in num_players), \
        "number of players must be greater than zero"
    assert all(s > 0 for s in num_strategies), \
        "number of strategies must be greater than zero"

    # This list is necessary to maintain consistent order.
    roles = list(_random_strings(num_roles, prefix='r', cool=cool))
    strategies = {role: set(_random_strings(num_strat, prefix='s', cool=cool))
                  for role, num_strat
                  in zip(roles, num_strategies)}
    players = dict(zip(roles, num_players))
    return rsgame.EmptyGame(players, strategies)


def role_symmetric_game(num_roles, num_players, num_strategies,
                        distribution=default_distribution, cool=False):
    """Generate a random role symmetric game

    Parameters
    ----------
    num_roles : int > 0
        The number of roles in the game.
    num_players : int or [int], len == num_roles
        The number of players, same for each role if a scalar, or a list, one
        for each role.
    num_strategies : int or [int], len == num_roles
        The number of strategies, same for each role if a scalar, or a list,
        one for each role.
    distribution : () -> float
        Payoff distribution. Calling should result in a scalar payoff.
        (default: default_distribution)
    cool : bool
        Whether to generate word-like role and strategy strings. These will be
        random, and hence unpredictable, whereas standard role and strategy
        names are predictable. (default: False)
    """
    game = _gen_empty_rs_game(num_roles, num_players, num_strategies, cool)

    def profiles():
        for prof in game.all_profiles():
            payoffs = {role: {strat: distribution() for strat in strats}
                       for role, strats in prof.items()}
            yield prof.to_input_profile(payoffs)

    return rsgame.Game(game.players, game.strategies, profiles(), game.size)


def independent_game(num_players, num_strategies,
                     distribution=default_distribution, cool=False):
    """Generate a random independent (asymmetric) game

    All payoffs are generated independently from distribution.

    Parameters
    ----------
    num_players : int > 0
        The number of players.
    num_strategies : int or [int], len == num_players
        The number of strategies for each player. If an int, then every player
        has the same number of strategies.
    distribution : (shape) -> ndarray (shape)
        The distribution to sample payoffs from. Must take a single shape
        argument and return an ndarray of iid values with that shape.
    """
    num_strategies = _as_list(num_strategies, num_players)
    shape = num_strategies + [num_players]
    return _gen_matrix_game(distribution(shape), cool)


def symmetric_game(num_players, num_strategies,
                   distribution=default_distribution, cool=False):
    """Generate a random symmetric game"""
    return role_symmetric_game(1, num_players, num_strategies, distribution,
                               cool)


def covariant_game(num_players, num_strategies, mean_dist=lambda shape:
                   np.zeros(shape), var_dist=lambda shape: np.ones(shape),
                   covar_dist=default_distribution, cool=False):
    """Generate a covariant game

    Covariant games are asymmetric games where payoff values for each profile
    drawn according to multivariate normal.

    The multivariate normal for each profile has a constant mean drawn from
    `mean_dist`, constant variance drawn from`var_dist`, and constant
    covariance drawn from `covar_dist`.

    Parameters
    ----------
    mean_dist : (shape) -> ndarray (shape)
        Distribution from which mean payoff for each profile is drawn.
        (default: lambda: 0)
    var_dist : (shape) -> ndarray (shape)
        Distribution from which payoff variance for each profile is drawn.
        (default: lambda: 1)
    covar_dist : (shape) -> ndarray (shape)
        Distribution from which the value of the off-diagonal covariance matrix
        entries for each profile is drawn. (default: uniform [-1, 1])
    """
    num_strategies = _as_list(num_strategies, num_players)
    shape = num_strategies + [num_players]
    var = covar_dist(shape + [num_players])
    diag = var.diagonal(0, num_players, num_players + 1)
    diag.setflags(write=True)  # Hack
    np.copyto(diag, var_dist(shape))
    u, s, v = np.linalg.svd(var)
    # The next couple of lines do multivariate Gaussian sampling for all
    # payoffs simultaneously
    payoffs = rand.normal(size=shape)
    # FIXME Phrase a multiplication and sum to make more clear
    payoffs = np.einsum('...i,...ij->...j', payoffs, np.sqrt(s)[..., None] * v)
    payoffs += mean_dist(shape)
    return _gen_matrix_game(payoffs, cool)


def zero_sum_game(num_strategies, distribution=default_distribution,
                  cool=False):
    """Generate a two-player, zero-sum game"""
    p1_payoffs = distribution([num_strategies, num_strategies])
    return _gen_matrix_game(np.dstack([p1_payoffs, -p1_payoffs]), cool)


def sym_2p2s_game(a=0, b=1, c=2, d=3, distribution=default_distribution,
                  cool=False):
    """Create a symmetric 2-player 2-strategy game of the specified form.

    Four payoff values get drawn from U(min_val, max_val), and then are
    assigned to profiles in order from smallest to largest according to the
    order parameters as follows:

       | s0  | s1  |
    ---|-----|-----|
    s0 | a,a | b,c |
    s1 | c,b | d,d |
    ---|-----|-----|

    So a=2,b=0,c=3,d=1 gives a prisoners' dilemma; a=0,b=3,c=1,d=2 gives a game
    of chicken.

    distribution must accept a size parameter a la numpy distributions."""
    game = _gen_empty_rs_game(1, 2, 2, cool=cool)
    role, strats = next(iter(game.strategies.items()))
    strats = list(strats)

    payoffs = sorted(distribution(4))
    profile_data = [
        {role: [(strats[0], 2, [payoffs[a]])]},
        {role: [(strats[0], 1, [payoffs[b]]),
                (strats[1], 1, [payoffs[c]])]},
        {role: [(strats[1], 2, [payoffs[d]])]}]
    return rsgame.Game(game.players, game.strategies, profile_data)


# FIXME add game constructor with payoffs and values and switch all games to
# use it
def congestion_game(num_players, num_facilities, num_required, cool=False):
    """Generates a random congestion game with num_players players and nCr(f, r)
    strategies

    Congestion games are symmetric, so all players belong to one role. Each
    strategy is a subset of size #required among the size #facilities set of
    available facilities. Payoffs for each strategy are summed over facilities.
    Each facility's payoff consists of three components:

    -constant ~ U[0, num_facilities]
    -linear congestion cost ~ U[-num_required, 0]
    -quadratic congestion cost ~ U[-1, 0]
    """
    # Generate strategies mask
    strat_list = list(itertools.combinations(range(num_facilities),
                                             num_required))
    num_strats = len(strat_list)
    num_strats = scm.comb(num_facilities, num_required, exact=True)
    strat_mask = np.zeros([num_strats, num_facilities], dtype=bool)
    inds = np.fromiter(
        itertools.chain.from_iterable(
            (row * num_facilities + f for f in facs) for row, facs
            in enumerate(strat_list)),
        int, num_strats * num_required)
    strat_mask.ravel()[inds] = True

    # Generate value for congestions
    values = rand.random((num_facilities, 3))
    values[:, 0] *= num_facilities  # constant
    values[:, 1] *= -num_required   # linear
    values[:, 2] *= -1              # quadratic

    # Compute array version of all payoffs
    counts = _symmetric_counts(num_players, num_strats)

    # Compute usage of every facility and then payoff
    strat_usage = counts[..., None] * strat_mask
    usage = strat_usage.sum(1)
    fac_payoffs = (usage[..., None] ** np.arange(3) * values).sum(2)
    payoffs = (strat_usage * fac_payoffs[:, None, :]).sum(2)

    # Generate names for everything
    strategies = (list(_random_strings(num_strats, cool=cool)) if cool
                  else ['_'.join(str(s) for s in strat)
                        for strat in strat_list])
    role = next(_random_strings(num_strats, cool=cool)) if cool else 'all'

    # Generator of profile data
    profiles = ({role: [(strategies[i], c, p) for i, (c, p)
                        in enumerate(zip(count, payoff)) if c > 0]}
                for count, payoff in zip(counts, payoffs))
    return rsgame.Game({role: num_players}, {role: strategies}, profiles,
                       counts.shape[0])


def local_effect_game(num_players, num_strategies, cool=False):
    """Generates random congestion games with num_players (N) players and
    num_strategies (S) strategies.

    Local effect games are symmetric, so all players belong to one role. Each
    strategy corresponds to a node in the G(N, 2/S) (directed edros-renyi
    random graph with edge probability of 2/S) local effect graph. Payoffs for
    each strategy consist of constant terms for each strategy, and interaction
    terms for the number of players choosing that strategy and each neighboring
    strategy.

    The one-strategy terms are drawn as follows:
    -constant ~ U[-(N+S), N+S]
    -linear ~ U[-N, 0]

    The neighbor strategy terms are drawn as follows:
    -linear ~ U[-S, S]
    -quadratic ~ U[-1, 1]

    """
    # Generate local effects graph. This is an SxSx3 graph where the first two
    # axis are in and out nodes, and the final axis is constant, linear,
    # quadratic gains.

    # There's a little redundant computation here (what?)
    local_effects = np.empty((num_strategies, num_strategies, 3))
    # Fill in neighbors
    local_effects[..., 0] = 0
    local_effects[..., 1] = rand.uniform(-num_strategies, num_strategies,
                                         (num_strategies, num_strategies))
    local_effects[..., 2] = rand.uniform(-1, 1,
                                         (num_strategies, num_strategies))
    # Mask out some edges
    local_effects *= (rand.random((num_strategies, num_strategies)) >
                      (2 / num_strategies))[..., np.newaxis]
    # Fill in self
    np.fill_diagonal(local_effects[..., 0],
                     rand.uniform(-(num_players + num_strategies),
                                  num_players + num_strategies,
                                  num_strategies))
    np.fill_diagonal(local_effects[..., 1],
                     rand.uniform(-num_players, 0, num_strategies))
    np.fill_diagonal(local_effects[..., 2], 0)

    # Compute all profiles and payoffs
    counts = _symmetric_counts(num_players, num_strategies)
    payoffs = (local_effects * counts[..., None, None] ** np.arange(3))\
        .sum((1, 3))

    # Compute string names of things
    role = next(_random_strings(1, prefix='r', cool=cool))
    strategies = list(_random_strings(num_strategies, prefix='s', cool=cool))

    # Generate input profiles
    profiles = ({role: [(strategies[i], c, p) for i, (c, p)
                        in enumerate(zip(count, payoff)) if c > 0]}
                for count, payoff in zip(counts, payoffs))

    return rsgame.Game({role: num_players}, {role: strategies}, profiles,
                       counts.shape[0])


# TODO make more efficient i.e. don't loop through all player combinations in
# python
# TODO make matrix game generate the compact form instead of calling the
# function on it...
def polymatrix_game(num_players, num_strategies, matrix_game=independent_game,
                    players_per_matrix=2, cool=False):
    """Creates a polymatrix game using the specified k-player matrix game function.

    Each player's payoff in each profile is a sum over independent games played
    against each set of opponents. Each k-tuple of players plays an instance of
    the specified random k-player matrix game.

    players_per_matrix: k
    matrix_game:        a function of two arguments (player_per_matrix,
                        num_strategies) that returns 2-player,
                        num_strategies-strategy games.

    Note: The actual roles and strategies of matrix game are ignored.
    """
    payoffs = np.zeros([num_strategies] * num_players + [num_players])
    for players in itertools.combinations(range(num_players),
                                          players_per_matrix):
        subgame = matrix_game(players_per_matrix, num_strategies)
        sub_payoffs, _ = _compact_payoffs(subgame)
        new_shape = np.array([1] * num_players + [players_per_matrix])
        new_shape[list(players)] = num_strategies
        payoffs[..., list(players)] += sub_payoffs.reshape(new_shape)

    return _gen_matrix_game(payoffs, cool)


def add_noise(game, num_samples, noise=default_distribution):
    """Generate sample game by adding noise to game payoffs

    Arguments
    ---------
    game:        A Game or SampleGame (only current payoffs are used)
    num_samples: The number of observations to create per profile
    noise:       A noise generating function. The function should take a single
                 shape parameter, and return a number of samples equal to
                 shape. In order to preserve mixed equilibria, noise should
                 also be zero mean (aka unbiased)
    """
    def profiles():
        for counts, values in game.payoffs(as_array=True):
            new_values = (values[..., np.newaxis] +
                          noise(values.shape + (num_samples,)))
            prof = game.as_profile(counts)
            payoffs = game._payoff_dict(counts, new_values,
                                        lambda l: list(map(float, l)))
            yield prof.to_input_profile(payoffs)
    return rsgame.SampleGame(game.players, game.strategies, profiles(),
                             len(game))


# def gaussian_mixture_noise(max_stdev, samples, modes=2, spread_mult=2):
#     """
#     Generate Gaussian mixture noise to add to one payoff in a game.

#     max_stdev: maximum standard deviation for the mixed distributions (also
#                 affects how widely the mixed distributions are spaced)
#     samples: numer of samples to take of every profile
#     modes: number of Gaussians to mix
#     spread_mult: multiplier for the spread of the Gaussians. Distance between
#                 the mean and the nearest distribution is drawn from
#                 N(0,max_stdev*spread_mult).
#     """
#     multipliers = arange(float(modes)) - float(modes-1)/2
#     offset = normal(0, max_stdev * spread_mult)
#     stdev = beta(2,1) * max_stdev
#     return [normal(choice(multipliers)*offset, stdev) for _ in range(samples)] # noqa


# eq_var_normal_noise = partial(normal, 0)
# normal_noise = partial(gaussian_mixture_noise, modes=1)
# bimodal_noise = partial(gaussian_mixture_noise, modes=2)


# def nonzero_gaussian_noise(max_stdev, samples, prob_pos=0.5, spread_mult=1):
#     """
#     Generate Noise from a normal distribution centered up to one stdev from 0. # noqa

#     With prob_pos=0.5, this implements the previous buggy output of
#     bimodal_noise.

#     max_stdev: maximum standard deviation for the mixed distributions (also
#                 affects how widely the mixed distributions are spaced)
#     samples: numer of samples to take of every profile
#     prob_pos: the probability that the noise mean for any payoff will be >0.
#     spread_mult: multiplier for the spread of the Gaussians. Distance between
#                 the mean and the mean of the distribution is drawn from
#                 N(0,max_stdev*spread_mult).
#     """
#     offset = normal(0, max_stdev)*(1 if U(0,1) < prob_pos else -1)*spread_mult # noqa
#     stdev = beta(2,1) * max_stdev
#     return normal(offset, stdev, samples)


# def uniform_noise(max_half_width, samples):
#     """
#     Generate uniform random noise to add to one payoff in a game.

#     max_range: maximum half-width of the uniform distribution
#     samples: numer of samples to take of every profile
#     """
#     hw = beta(2,1) * max_half_width
#     return U(-hw, hw, samples)


# def gumbel_noise(scale, samples, flip_prob=0.5):
#     """
#     Generate random noise according to a gumbel distribution.

#     Gumbel distributions are skewed, so the default setting of the flip_prob
#     parameter makes it equally likely to be skewed positive or negative

#     variance ~= 1.6*scale
#     """
#     location = -0.5772*scale
#     multiplier = -1 if (U(0,1) < flip_prob) else 1
#     return multiplier * gumbel(location, scale, samples)


# def mix_models(models, rates, spread, samples):
#     """
#     Generate SampleGame with noise drawn from several models.

#     models: a list of 2-parameter noise functions to draw from
#     rates: the probabilites with which a payoff will be drawn from each model
#     spread, samples: the parameters passed to the noise functions
#     """
#     cum_rates = cumsum(rates)
#     m = models[bisect(cum_rates, U(0,1))]
#     return m(spread, samples)


# n80b20_noise = partial(mix_models, [normal_noise, bimodal_noise], [.8,.2])
# n60b40_noise = partial(mix_models, [normal_noise, bimodal_noise], [.6,.4])
# n40b60_noise = partial(mix_models, [normal_noise, bimodal_noise], [.4,.6])
# n20b80_noise = partial(mix_models, [normal_noise, bimodal_noise], [.2,.8])

# equal_mix_noise = partial(mix_models, [normal_noise, bimodal_noise, \
#         uniform_noise, gumbel_noise], [.25]*4)
# mostly_normal_noise =  partial(mix_models, [normal_noise, bimodal_noise, \
#         gumbel_noise], [.8,.1,.1])

# noise_functions = filter(lambda k: k.endswith("_noise") and not \
#                     k.startswith("add_"), globals().keys())

# def rescale_payoffs(game, min_payoff=0, max_payoff=100):
#     """
#     Rescale game's payoffs to be in the range [min_payoff, max_payoff].

#     Modifies game.values in-place.
#     """
#     game.makeArrays()
#     min_val = game.values.min()
#     max_val = game.values.max()
#     game.values -= min_val
#     game.values *= (max_payoff - min_payoff)
#     game.values /= (max_val - min_val)
#     game.values += min_payoff
