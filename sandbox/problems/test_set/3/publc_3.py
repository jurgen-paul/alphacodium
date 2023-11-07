def run():
    def check(s, i):
        if i == len(s):
            x = ''.join(s)
            if x[0] == '0' and x != '0':
                return 0
            if int(x) % 25 == 0:
                return 1
            return 0
        res = 0
        if s[i] == '_':
            if i != 0 and i < len(s) - 2:
                s[i] = '1'
                res += 10 * check(s, i + 1)
            else:
                for j in range(10):
                    s[i] = str(j)
                    res += check(s, i + 1)
            s[i] = '_'
            return res
        else:
            return check(s, i + 1)

    inp = input()
    total = 0
    if 'X' in inp:
        for i in range(10):
            s = inp
            s = s.replace('X', str(i))
            total += check(list(s), 0)
    else:
        total = check(list(inp), 0)

    print(total)

# # public test
inputs  = ['_XX']
outputs = ['9']
#
# # private tests:
# inputs = ['2 1\n-1000000000 1000000000\n2 1\n2 1 2\n',
#  '4 4\n2 -1000 100 3\n2 1\n3 2\n4 1\n2 1 3\n2 2 2\n1 1 -1000000000\n2 1 4\n']
# outputs = ['2000000000\n', '2102\n0\n1000000003\n']


import io
from alpha_codium.code_contests.eval.local_exec import swallow_io

input_stream = io.BytesIO(inputs[0].encode())
input_stream.seek(0)
try:
    with swallow_io(input_stream=input_stream) as (stdout_stream, stderr_stream):
        run()
    print(stdout_stream.getvalue())
except Exception as E:
    print(E)
    pass