"""sample profiles from a mixture"""
import argparse
import itertools
import json
import sys
from os import path

from gameanalysis import gameio


PACKAGE = path.splitext(path.basename(sys.modules[__name__].__file__))[0]
PARSER = argparse.ArgumentParser(prog='ga ' + PACKAGE, description="""Sample
                                 profiles from a mixture.""")
PARSER.add_argument('--input', '-i', metavar='<input-file>', default=sys.stdin,
                    type=argparse.FileType('r'), help="""Game file to draw
                    samples from.  (default: stdin)""")
PARSER.add_argument('--mix', '-m', metavar='<mixture-file>', default=sys.stdin,
                    type=argparse.FileType('r'), help="""Mixture to sample
                    profiles from. (default: stdin)""")
PARSER.add_argument('--output', '-o', metavar='<output-file>',
                    default=sys.stdout, type=argparse.FileType('w'),
                    help="""File to write stream of profiles too. (default:
                    stdout)""")
PARSER.add_argument('--num', '-n', metavar='<num-samples>', default=1,
                    type=int, help="""The number of samples to gather.
                    (default: %(default)d)""")
PARSER.add_argument('--deviations', '-d', action='store_true', help="""Sample
                    deviation profiles instead of general mixture profiles.
                    This will sample `num` profiles from each deviation. The
                    output will be a stream of json objects with a devrole,
                    devstrat, and profile field.""")


def main():
    args = PARSER.parse_args()
    game, serial = gameio.read_game(json.load(args.input))
    mix = serial.from_prof_json(json.load(args.mix))

    if args.deviations:
        profs = (game.random_deviator_profiles(mix, args.num)
                 .reshape((-1, game.num_role_strats)))
        dev_names = itertools.cycle(itertools.chain.from_iterable(
            ((r, s) for s in ses) for r, ses
            in zip(serial.role_names, serial.strat_names)))
        for prof, (devrole, devstrat) in zip(profs, dev_names):
            json.dump(dict(
                devrole=devrole,
                devstrat=devstrat,
                profile=serial.to_prof_json(prof)), args.output)
            args.output.write('\n')

    else:
        for prof in game.random_profiles(mix, args.num):
            json.dump(serial.to_prof_json(prof), args.output)
            args.output.write('\n')


if __name__ == '__main__':
    main()
