from gameanalysis import AGGFN
from gameanalysis import AGGgen
from gameanalysis import gpgame
import numpy as np

TINY = float(np.finfo(float).tiny)

num_players = 7
num_facilities = 4
num_required = 2

################# Generate random congestion game ##################
game, json = AGGgen.congestion_game(num_players, num_facilities, num_required)
rs = game.to_rsgame()
gp = gpgame.BaseGPGame(rs)

def calc_var(orig, learned):
    """
    calculates the error rate of the learned payoffs
    Input:
        - orig: a 2D array of payoffs
        - learned: the learned payoffs corresponding to the same profiles
    Return:
        - rate of error
    """
    num_payoffs = np.prod(orig.shape)
    error = np.absolute(
            (1 - (learned.ravel()+ TINY) / (orig.ravel() + TINY))).sum()
    return error / num_payoffs

def compute_error(game, gp):
    mixture = game.uniform_mixture()
    profs = game.random_profiles(mixture, num_samples=100)
    gp_payoffs = gp.get_payoffs(profs)
    rs_payoffs = np.array([rs.get_payoffs(prof) for prof in profs])
    return calc_var(gp_payoffs, rs_payoffs)

print(compute_error(game, gp))
