from urllib import urlopen
from json import loads
from xml.dom.minidom import parseString, Document
from os.path import exists, splitext

from RoleSymmetricGame import *


def readGame(source):
	if isinstance(source, str) and exists(source):
		#source is a filename
		f = open(source)
		data = f.read()
		f.close()
	elif isinstance(source, str) and source.startswith("http://"):
		#source is a url
		u = urlopen(source)
		data = u.read()
		u.close()
	elif hasattr(source, 'read'):
		#source is file-like
		data = source.read()
		source.close()
	else:
		#assume source is already xml or json data
		data = source
	try:
		return readJSON(data)
	except:
		try:
			return readXML(data)
		except:
			raise IOError("invalid game source")


def readJSON(data):
	if isinstance(data, str):
		json_data = loads(data)
	else:
		json_data = data
	if "strategy_array" in json_data["roles"][0]:
		return readJSON_old(json_data)
	elif "strategies" in json_data["roles"][0]:
		return readJSON_v2(json_data)
	else:
		raise IOError("invalid JSON data: " + str(json_data))


def readJSON_v2(json_data):
	counts = {r["name"] : int(r["count"]) for r in json_data["roles"]}
	strategies = {r["name"] : r["strategies"] for r in json_data["roles"]}
	roles = list(counts.keys())
	payoffs = []
	if "profiles" in json_data:
		for profileDict in json_data["profiles"]:
			profile = {r:[] for r in roles}
			prof_strat = {}
			for roleDict in profileDict["roles"]:
				role = roleDict["name"]
				for strategyDict in roleDict["strategies"]:
					profile[role].append(payoff_data(str(strategyDict[ \
							"name"]), int(strategyDict["count"]), \
							float(strategyDict["payoff"])))
			payoffs.append(profile)
	return Game(roles, counts, strategies, payoffs)


def readJSON_old(json_data):
	counts = {r["name"] : int(r["count"]) for r in json_data["roles"]}
	strategies = {r["name"] : r["strategy_array"] for r in json_data["roles"]}
	roles = list(counts.keys())
	payoffs = []
	for profileDict in json_data["profiles"]:
		profile = {r:[] for r in roles}
		prof_strat = {}
		for role_str in profileDict["proto_string"].split("; "):
			role, strategy_str = role_str.split(": ")
			prof_strat[role] = strategy_str.split(", ")
		for roleDict in profileDict["roles"]:
			role = roleDict["name"]
			for strategyDict in roleDict["strategies"]:
				s = strategyDict["name"]
				profile[role].append(payoff_data(str(s), int(prof_strat[role] \
						.count(s)), float(strategyDict["payoff"])))
		payoffs.append(profile)
	return Game(roles, counts, strategies, payoffs)


def readXML(data):
	if isinstance(data, str):
		xml_data = parseString(data)
	else:
		xml_data = data
	gameNode = xml_data.getElementsByTagName("nfg")[0]
	if len(gameNode.getElementsByTagName("player")[0]. \
			getElementsByTagName("action")) > 0:
		return parseStrategicXML(gameNode)
	return parseSymmetricXML(gameNode)


def parseStrategicXML(gameNode):
	strategies = {p.getAttribute('id') : map(lambda s: s.getAttribute('id'), \
			p.getElementsByTagName('action')) for p in \
			gameNode.getElementsByTagName('player')}
	roles = list(strategies.keys())
	counts = {r:1 for r in roles}
	payoffs = []
	for payoffNode in gameNode.getElementsByTagName('payoff'):
		data = {r:[] for r in roles}
		for outcomeNode in payoffNode.getElementsByTagName('outcome'):
			role = outcomeNode.getAttribute('player')
			strategy = outcomeNode.getAttribute('action')
			value = float(outcomeNode.getAttribute('value'))
			data[role].append(payoff_data(strategy, 1, value))
		payoffs.append(data)
	return Game(roles, counts, strategies, payoffs)


def parseSymmetricXML(gameNode):
	roles = ["All"]
	counts= {"All" : len(gameNode.getElementsByTagName("player"))}
	strategies = {"All" : [e.getAttribute("id") for e in \
			gameNode.getElementsByTagName("action")]}
	payoffs = []
	for payoffNode in gameNode.getElementsByTagName("payoff"):
		data = []
		for outcomeNode in payoffNode.getElementsByTagName("outcome"):
			strategy = outcomeNode.getAttribute("action")
			count = int(outcomeNode.getAttribute("count"))
			value = float(outcomeNode.getAttribute("value"))
			data.append(payoff_data(strategy, count, value))
		payoffs.append({"All":data})
	return Game(roles, counts, strategies, payoffs)


def toJSON(game, **other_data):
	"""
	Writes game to JSON according to the Testbed role-symmetric game spec.
	"""
	game_dict = {}
	game_dict["roles"] = [{"name":role, "count":game.players[role], \
				"strategies": game.strategies[role]} for role in game.roles]
	game_dict["profiles"] = [{"roles":[{"name":role, "strategies":["TODO"]} for role in game.roles]} for profile in game]
	raise NotImplementedError("TODO")


def writeXML(game, filename):
	if len(game.roles) == 1:
		writeSymmetricXML(game, filename)
	elif all(map(lambda c: c==1, game.counts.values())):
		writeStrategicXML(game, filename)
	else:
		raise NotImplementedError("no EGAT XML spec for role-symmetric games")


def writeSymmetricXML(game, filename):
	"""
	Writes game to XML according to the EGAT symmetric game spec.
	Assumes (but doesn't check) that game is symmetric.
	"""
	raise NotImplementedError("TODO")

def writeStrategicXML(game, filename):
	"""
	Writes game to XML according to the EGAT strategic game spec.
	Assumes (but doesn't check) that game is not role-symmetric.
	"""
	raise NotImplementedError("TODO")

