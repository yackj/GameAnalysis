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

    return np.array([ MAE(gp_payoffs, game_payoffs), 
                      MAPE(gp_payoffs, game_payoffs),
                      SMAPE(gp_payoffs, game_payoffs),
                      float(rs.max_payoffs() - rs.min_payoffs()),
                      float(orig.max_payoffs() - orig.min_payoffs()) ])


def get_pure_accuracy(num_players, num_facilities, num_required, prop=1, noise=None,
                 samples=10, num_reps=5, comp_orig=True):
    """
    Runs an reduction and gets an approximate accuracy of the reduction
    """
    # results is an array of 5 elements: mae, mape, smape, rs_gap, orig_gap
    results = np.zeros(5, dtype=float)

    for rep in range(num_reps):
        game, rs, json = generate(num_players, num_facilities, num_required,
                                  prop=prop, noise=noise, samples=samples)
        gp = gpgame.BaseGPGame(rs)
        results = results + compute_pure_error(rs, gp, game, comp_orig=comp_orig)

    return results / num_reps


def compute_mixed_error(rs, gp, orig, use_all=False, num_samples=1000, \
                        comp_orig=True, method=None):
    if method not in ['DPR', 'Point', 'Neighbor', 'Sample']:
        raise ValueError('Reduction Method illegal')
    profs = rs.random_mixtures(num_samples=num_samples)
    


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

