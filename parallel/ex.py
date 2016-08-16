from gameanalysis import AGGvalidation as AGGV
import sys
import json

num = sys.argv[1]

f_in = open('./{}.in'.format(num))
f_out = open('./{}.out'.format(num), 'w')

for line in f_in:
    params = eval(line)
    pure, mix = AGGV.accuracy_experiment(*params)
    f_out.write( json.dump( dict(
                ('pure', pure.tolist()), ('mix', mix.tolist()) )))
    f_out.write('\n')
