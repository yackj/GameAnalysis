from gameanalysis import AGGFN
from gameanalysis import AGGgen
from gameanalysis import gpgame
from random import choice

num_players = 7
num_facilities = 4
num_required = 2

# Generate random congestion game
game, json = AGGgen.congestion_game(num_players, num_facilities, num_required)
rs = game.to_rsgame()
gp = gpgame.BaseGPGame(rs)

# check different in pure strategy profile payoffs
prof = choice(game.all_profiles())
