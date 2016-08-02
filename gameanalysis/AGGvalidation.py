from gameanalysis import AGGFN
from gameanalysis import AGGgen
from gameanalysis import gpgame
import numpy as np
import matplotlib.pyplot as plt

TINY = float(np.finfo(float).tiny)

game_gen_params = [6, 5, 2]

def generate(num_players, num_facilities, num_required, prop=1):
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
    return game


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
    num_payoffs = np.prod(orig.shape)
    error = np.absolute(
            (1 - (learned + TINY) / (orig + TINY))).sum()
    return error / num_payoffs


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
    num_payoffs = np.prod(orig.shape)
    error = np.absolute(learned - orig).sum()
    return error / num_payoffs


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
    num_payoffs = np.prod(orig.shape)
    error = ((np.absolute(orig - learned) + TINY) / \
        (np.absolute(learned) + np.absolute(orig) + TINY))
    error = error[orig != 0].sum()
    return error / num_payoffs


def compute_error(game, gp):
    mixture = game.uniform_mixture()
    profs = game.random_profiles(mixture, num_samples=1000)
    profs = np.array([prof for prof in profs if prof in game])
    gp_payoffs = gp.get_payoffs(profs)
    game_payoffs = np.array([game.get_payoffs(prof) for prof in profs])
    #return MAPE(gp_payoffs, game_payoffs)
    return MAE(gp_payoffs, game_payoffs)
    #return SMAPE(gp_payoffs, game_payoffs)


def decay_games(game, num_rep=30, num_steps=20):
    error = np.empty([num_rep, num_steps], dtype=float)
    for rep in range(num_rep):
        for i in range(num_steps): 
            rs = game.to_rsgame(prop=i*(0.9/num_steps)+0.1)
            gp = gpgame.BaseGPGame(rs)
            error[rep][i] = compute_error(rs, gp)
    error = error.sum(0) / num_rep
    return error


game = generate(*game_gen_params)
e = decay_games(game, num_rep=20)
