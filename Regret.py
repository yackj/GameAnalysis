#! /usr/bin/env python2.7

from itertools import product, chain, combinations_with_replacement as CwR

import numpy as np

from RoleSymmetricGame import *


def regret(game, prof, role=None, strategy=None, deviation=None, bound=False):
	if role == None and len(game.roles) == 1:
		role = game.roles[0] #in symmetric games we can deduce the role
	if is_pure_profile(prof):
		return profile_regret(game, prof, role, strategy, deviation, bound)
	if is_mixture_array(prof):
		return mixture_regret(game, prof, role, deviation, bound)
	if is_mixed_profile(prof):
		return mixture_regret(game, game.toArray(prof), role, deviation, bound)
	if is_profile_array(prof):
		return profile_regret(game, game.toProfile(prof), role, strategy, \
				deviation, bound)
	raise TypeError(one_line("unrecognized profile type: " + str(prof), 69))


def profile_regret(game, prof, role, strategy, deviation, bound):
	if role == None:
		return max([profile_regret(game, prof, r, strategy, deviation, bound) \
				for r in game.roles])
	if strategy == None:
		return max([profile_regret(game, prof, role, s, deviation, bound) for \
				s in prof[role]])
	if deviation == None:
		try:
			return max([profile_regret(game, prof, role, strategy, d, bound) \
					for d in set(game.strategies[role]) - {strategy}])
		except ValueError: #triggered when there's only one strategy
			return 0
	dev_prof = prof.deviate(role, strategy, deviation)
	if dev_prof in game:
		return game.getPayoff(dev_prof, role, deviation) - \
				game.getPayoff(prof, role, strategy)
	else:
		return -float("inf") if bound else float("inf")


def mixture_regret(game, mix, role, deviation, bound):
	strategy_EVs = game.expectedValues(mix)
	role_EVs = (strategy_EVs * mix).sum(1)
	if role == None:
		return max([float(max(strategy_EVs[r][:game.numStrategies[r]]) - \
			role_EVs[r]) for r in range(len(game.roles))])
	r = game.index(role)
	if deviation == None:
		return float(max(strategy_EVs[r][:game.numStrategies[r]]) - \
			role_EVs[r])
	if any(map(lambda p: p not in game, mixture_neighbors(game, \
			mix, role, deviation))):
		return -float("inf") if bound else float("inf")
	d = game.index(role, deviation)
	return float(strategy_EVs[r,d] - role_EVs[r])


def neighbors(game, p, *args, **kwargs):
	if isinstance(p, Profile):
		return profile_neighbors(game, p, *args, **kwargs)
	elif isinstance(p, np.ndarray):
		return mixture_neighbors(game, p, *args, **kwargs)
	raise TypeError("unrecognized argument type: " + type(p).__name__)


def profile_neighbors(game, profile, role=None, strategy=None, \
		deviation=None):
	if role == None:
		return list(chain(*[profile_neighbors(game, profile, r, strategy, \
				deviation) for r in game.roles]))
	if strategy == None:
		return list(chain(*[profile_neighbors(game, profile, role, s, \
				deviation) for s in profile[role]]))
	if deviation == None:
		return list(chain(*[profile_neighbors(game, profile, role, strategy, \
				d) for d in set(game.strategies[role]) - {strategy}]))
	return [profile.deviate(role, strategy, deviation)]


def mixture_neighbors(game, mix, role=None, deviation=None):
	n = set()
	for profile in feasible_profiles(game, mix):
		n.update(profile_neighbors(game, profile, role, deviation=deviation))
	return n


def feasible_profiles(game, mix, thresh=1e-3):
	return [Profile({r:{s:p[game.index(r)].count(s) for s in set(p[ \
			game.index(r)])} for r in game.roles}) for p in product(*[ \
			CwR(filter(lambda s: mix[game.index(r), game.index(r,s)] >= \
			thresh, game.strategies[r]), game.players[r]) for r \
			in game.roles])]


def symmetric_profile_regrets(game):
	assert len(game.roles) == 1, "game must be symmetric"
	role = game.roles[0]
	return {s: regret(game, Profile({role:{s:game.players[role]}})) for s \
			in game.strategies[role]}


def equilibrium_regrets(game, eq):
	regrets = {}
	for role in game.roles:
		regrets[role] = {}
		for strategy in game.strategies[role]:
			regrets[role][strategy] = -regret(game, eq, deviation=strategy)
	return regrets


from GameIO import read, to_JSON_str, io_parser
from Subgames import translate

def parse_args():
	parser = io_parser(description="Compute regret in input game(s) of " +\
			"specified profiles.")
	parser.add_argument("profiles", type=str, help="File with profiles from" +\
			" input games for which regrets should be calculated.")
	return parser.parse_args()


def main():
	args = parse_args()
	games = args.input
	profiles = read(args.profiles)
	if not isinstance(profiles, list):
		profiles = [profiles]
	if not isinstance(games, list):
		games = [games] * len(profiles)
	regrets = []
	for g, prof_list in zip(games, profiles):
		regrets.append([])
		for prof in prof_list:
			regrets[-1].append(regret(g, prof))
	if len(regrets) > 1:
		print to_JSON_str(regrets)
	else:
		print to_JSON_str(regrets[0])


if __name__ == "__main__":
	main()

