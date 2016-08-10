from gameanalysis import AGGvalidation as AGGV
import sys

num = sys.argv[1]

f_in = open('./{}.in'.format(num))
f_out = open('./{}.out'.format(num), 'w')

for line in f_in:
    params = eval(line)
    ret = AGGV.get_accuracy(*params)
    f_out.write(str(list(ret))+'\n')
