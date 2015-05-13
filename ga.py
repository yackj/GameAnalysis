#! /usr/bin/env python3
import sys
import argparse
import nose
import collections
import textwrap

from gameanalysis import randgames

# TODO add a way to run tests from here


class HelpPrinter(object):
    '''Handles printing help with the help command'''

    @staticmethod
    def command(args, prog, print_help=False):
        parser = argparse.ArgumentParser(description='''Get help on various
        game analysis functions.''', add_help=False)
        parser.prog = ('%s %s' % (parser.prog, prog))
        parser.add_argument('command', nargs='?', choices=COMMANDS.keys(),
                            help='''The command to get help on. Leave empty to
                            get help on the analysis script.''')
        if print_help:
            parser.print_help()
            return
        args = parser.parse_args(args)
        if args.command is None:
            PARSER.print_help()
        else:
            COMMANDS[args.command].command([], args.command, True)


class TestRunner(object):
    '''Handles printing help with the help command'''

    @staticmethod
    def command(args, prog, print_help=False):
        parser = argparse.ArgumentParser(description='''Run game analysis tests
        to make sure everything is working properly. Any extra arguments passed
        in will be forwarded to nose. See the documentation of nosetests.''',
                                         add_help=False)
        parser.prog = ('%s %s' % (parser.prog, prog))
        parser.add_argument('--big-tests', '-b', action='store_true',
                            help='''Run the larger tests. This may fail if your
                            machine doesn't have enough memory.''')
        if print_help:
            parser.print_help()
            return
        args, extra = parser.parse_known_args(args)

        try:
            import test
        except ImportError:
            sys.stderr.write(textwrap.fill('''For some reason the test module
            cannot be imported. You may not have it, or there may be some other
            error causing test to not be imported. Try diagnosing with
            `nosetests -w test` in the root directory.''', 80))
            sys.stderr.write('\n')
            sys.exit(1)

        test.config(big_tests=args.big_tests)
        nose.run(module=test, argv=[parser.prog] + extra)


COMMANDS = collections.OrderedDict([
    ('rand', randgames),
    ('test', TestRunner),
    ('help', HelpPrinter)
])

# TODO: Help should probably have the main parser. Instead, this should look
# for the first argument to parse, and if failing that, call help by default
# with the appropriate arguments. OR something...

PARSER = argparse.ArgumentParser(description='''This script is way you call all
game analysis functions''', add_help=False)
PARSER.add_argument('command', choices=COMMANDS.keys(),
                    help='''The game analysis function to run''')


def main():
    args, extra = PARSER.parse_known_args()
    COMMANDS[args.command].command(extra, args.command)

if __name__ == '__main__':
    main()
