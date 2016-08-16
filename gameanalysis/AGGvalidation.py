from gameanalysis import AGGFN
from gameanalysis import AGGgen
from gameanalysis import gpgame
from gameanalysis import reduction
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

TINY = float(np.finfo(float).tiny)

game_gen_params = [6, 5, 2]

def generate(num_players, num_facilities, num_required, prop=1, noise=None,
             samples=10):
    """
    generate a random congestion game using the given params, and train
    a GP game with profiles from the generated congestion game
    Parameters:
        - prop: the proportion of profiles to use in training
    Return:
        - game: AGGFNA representation of the congestion game
        - gp: the learned Gaussian Process representation of the game
    """
    game, json = AGGgen.congestion_game(
                    num_players, num_facilities, num_required)
    rs = game.to_rsgame(prop=prop, noise=noise, samples=samples)
    return game, rs, json


def MAPE(orig, learned):
    """
    calculates the error rate of the learned payoffs
    MAPE: mean absolute percentage error
    Input:
        - orig: a 2D array of payoffs
        - learned: the learned payoffs corresponding to the same profiles
    Return:
        - MAPE error
    """
    return np.absolute(
            (1 - (learned + TINY) / (orig + TINY))).mean()


def MAE(orig, learned):
    """
    calculates the error rate of the learned payoffs
    MAE: mean absolute error
    Input:
        - orig: a 2D array of payoffs
        - learned: the learned payoffs corresponding to the same profiles
    Return:
        - MAE error
    """
    return np.absolute(learned - orig).mean()


def SMAPE(orig, learned):
    """
    calculates the error rate of the learned payoffs
    SMAPE: symmetric mean absolute percentage error
    Input:
        - orig: a 2D array of payoffs
        - learned: the learned payoffs corresponding to the same profiles
    Return:
        - SMAPE error
    """
    error = ((np.absolute(orig - learned) + TINY) / \
        (np.absolute(learned) + np.absolute(orig) + TINY))
    return error[orig != 0].mean()


def all_error(orig, learned):
    """
    Returns an array of MAE, MAPE, SMAPE measures of the original and learned
    payoffs
    """
    return np.array(
            [MAE(orig, learned), MAPE(orig, learned), SMAPE(orig, learned)] )


def compute_pure_error(rs, gp, orig, use_all=False, num_samples=1000, comp_orig=True):
    """
    This methods takes a gp game and computes its error by comparing to the
    learned pure strategy payoffs to those of the original game
    Parameters:
        - rs: the (reduced) rs game that is used in training of gp game
        - gp: the learned gp game representation of the original game
        - orig: the original game (either AGGFN or rsgame)
        - use_all: if use all the profiles for error calculation
        - num_samples: number of profiles sampled for error calculation when
                       not using the full game to calculate error
        - comp_orig: whether compare the learned payoff to original game or
                     the (reduced/noisy) rs game
    Return:
        - tuple of three different error metrics and two normalization factors
    """
    all_profs = rs.all_profiles()
    if use_all or len(all_profs) < num_samples:
        profs = all_profs
    else:
        profs = all_profs[rand.choice(all_profs.shape[0], num_samples, \
                                         replace=False)]
        profs = profs[np.array([prof in rs for prof in profs], dtype=bool)]
    gp_payoffs = gp.get_payoffs(profs)
    if comp_orig:
        game_payoffs = np.array([orig.get_payoffs(prof) for prof in profs])
    else:
        game_payoffs = np.array([rs.get_payoffs(prof) for prof in profs])

    return all_error(game_payoffs, gp_payoffs)


def compute_mixed_error(rs, gp, orig, use_all=False, num_samples=1000, \
                        comp_orig=True, method=None):
    """
    Computes the error of deviation payoffs of a reduced game
    Use all DPR, Point, Neighbor and Sample methods
    Return:
        - 2D array: |num_methods| row and |num_error_metrics| column
                    errors of all methods using all error metrics
    """
    profs = rs.random_mixtures(num_samples=num_samples)
    orig_pay = np.array([orig.deviation_payoffs(prof) for prof in profs])
    target_num_player = 5
    dpr = reduction.DeviationPreserving(orig.num_strategies[0],\
                                        orig.num_players[0],\
                                        target_num_players).reduce_game(agg.to_rsgame())
    dpr_pay = np.array([dpr.deviation_payoffs(prof) for prof in profs])
    pgp = gpgame.PointGPGame(bgp)
    pgp_pay = np.array([pgp.deviation_payoffs(prof) for prof in profs])
    sgp = gpgame.SampleGPGame(bgp)
    sgp_pay = np.array([sgp.deviation_payoffs(prof) for prof in profs])
    ngp = gpgame.NeighborGPGame(bgp)
    ngp_pay = np.array([ngp.deviation_payoffs(prof) for prof in profs])

    return np.array([all_error(orig_pay, dpr_pay),
                     all_error(orig_pay, pgp_pay),
                     all_error(orig_pay, sgp_pay),
                     all_error(orig_pay, ngp_pay) ])


def accuracy_experiment(num_players, num_facilities, num_required, prop=1, noise=None,
                 samples=10, num_reps=5, comp_orig=True):
    """
    Runs an reduction and gets an approximate accuracy of the reduction
    """
    # results is an array of 5 elements: mae, mape, smape, rs_gap, orig_gap
    pure_results = np.zeros(3, dtype=float)
    dev_results = np.zeros([4,3], dtype=float)

    for rep in range(num_reps):
        game, rs, json = generate(num_players, num_facilities, num_required,
                                  prop=prop, noise=noise, samples=samples)
        gp = gpgame.BaseGPGame(rs)
        pure_results = pure_results + compute_pure_error(rs, gp, game, comp_orig=comp_orig)
        dev_results = dev_results + compute_mixed_error(rs, gp, game, comp_orig=comp_orig)

    return pure_results / num_reps, dev_results / num_reps


def full_game_error(max_num_players, num_reps=5, noise=None, samples=10, cv_iter=8):
    """
    Validation experiment for training gp with the entire game
    """
    return np.array([[i, 3, 1, 1, noise, samples, reps, True] \
                    for i in range(3, max_num_players)],dtype=object)


def decay_games(size, num_reps=30, num_steps=20, noise=None, samples=10, cv_iter=8):
    return np.array([(size + [i*(0.95/num_steps)+0.05, noise, samples, num_reps, True]) \
            for i in range(num_steps)], dtype=object)


def noise_experiment(size, num_reps=30, max_noise=5, num_steps=20, comp_orig=False, cv_iter=8):
    return np.array([(size + [1, (0, i), 1, num_reps, comp_orig]) \
            for i in np.arange(0, max_noise, max_noise/num_steps)], dtype=object)


if __name__ == "__main__":
    get_accuracy(6, 5, 3, 1, None, samples=10, num_reps=1)

