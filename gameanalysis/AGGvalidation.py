from gameanalysis import AGGFN
from gameanalysis import AGGgen
from gameanalysis import gpgame
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

TINY = float(np.finfo(float).tiny)

game_gen_params = [6, 5, 2]

def generate(num_players, num_facilities, num_required, prop=1, noise=None,
             samples=10):
    #TODO: write game json to file
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


def compute_error(rs, gp, orig, use_all=False, num_samples=1000):
    """
    This methods takes a gp game and computes its error by comparing to the
    original game
    Parameters:
        - rs: the reduced rs game that is used in training of gp game
        - gp: the learned gp game representation of the original game
        - orig: the original game (either AGGFN or rsgame)
        - use_all: if use all the profiles for error calculation
        - num_samples: number of profiles sampled for error calculation when
                       not using the full game to calculate error
    Return:
        - tuple of three different error metrics and two normalization factors
    """
    all_profs = rs.all_profiles()
    if use_all or len(all_profs) < num_samples:
        profs = all_profs
    else:
        profs = all_profs[rand.choice(all_profs.shape[0], num_samples, \
                                         replace=False)]
    gp_payoffs = gp.get_payoffs(profs)
    game_payoffs = np.array([orig.get_payoffs(prof) for prof in profs])
    return ( MAE(gp_payoffs, game_payoffs), 
             MAPE(gp_payoffs, game_payoffs),
             SMAPE(gp_payoffs, game_payoffs),
             float(rs.max_payoffs() - rs.min_payoffs()),
             float(orig.max_payoffs() - orig.min_payoffs()) )


def full_game_error(max_num_players, reps, noise=None, samples=10):
    """
    Validation experiment for training gp with the entire game
    """
    mae, mape, smape, rs_gap, orig_gap = [], [], [], [], []

    for num_players in range(3, max_num_players):
        maer, maper, smaper, rs_gapr, orig_gapr = [], [], [], [], []
        for rep in range(reps):
            game, rs, json = generate(num_players, 3, 1,
                                      noise=noise, samples=samples)
            gp = gpgame.BaseGPGame(rs)
            ret = compute_error(rs, gp, game, use_all=True)

            maer.append(ret[0])
            maper.append(ret[1])
            smaper.append(ret[2])
            rs_gapr.append(ret[3])
            orig_gapr.append(ret[4])

        mae.append(sum(maer) / reps)
        mape.append(sum(maper) / reps)
        smape.append(sum(smaper) / reps)
        rs_gap.append(sum(rs_gapr) / reps)
        orig_gap.append(sum(orig_gapr) / reps)

    return mae, mape, smape, rs_gap, orig_gap


def decay_games(size, num_rep=30, num_steps=20, noise=None, samples=10):
    mae = np.empty([num_rep, num_steps], dtype=float)
    mape = np.empty([num_rep, num_steps], dtype=float)
    smape = np.empty([num_rep, num_steps], dtype=float)
    rs_gap = np.empty([num_rep, num_steps], dtype=float)
    orig_gap = np.empty([num_rep, num_steps], dtype=float)

    for rep in range(num_rep):
        for i in range(num_steps): 
            game, rs, json = generate(*size, prop=i*(0.95/num_steps)+0.05,
                                      noise=noise, samples=samples)
            gp = gpgame.BaseGPGame(rs)

            mae[rep][i], mape[rep][i], smape[rep][i], rs_gap[rep][i], \
                    orig_gap[rep][i] = compute_error(rs, gp, game)

    mae = mae.sum(0) / num_rep
    mape = mape.sum(0) / num_rep
    smape = smape.sum(0) / num_rep
    rs_gap = rs_gap.sum(0) / num_rep
    orig_gap = orig_gap.sum(0) / num_rep

    return mae, mape, smape, rs_gap, orig_gap


if __name__ == "__main__":
    #mae, mape, smape, rs_gap, orig_gap = full_game_error()
    mae, mape, smape, rs_gap, orig_gap = decay_games(game_gen_params, 1, 5)

