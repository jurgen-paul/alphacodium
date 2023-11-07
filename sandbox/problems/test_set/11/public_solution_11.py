def run():
    import sys
    sys.setrecursionlimit(100000)

    def _r():
        return sys.stdin.buffer.readline()

    def rs():
        return _r().decode('ascii').strip()

    def rn():
        return int(_r())

    def rnt():
        return map(int, _r().split())

    def rnl():
        return list(rnt())

    # [(2, -1), (1, 1), (4, -1), (2, 2), (5, 0), (3, 3), (7, 0)]
    # [(1, 1), (2, 2), (2, -1), (3, 3), (4, -1), (5, 0), (7, 0)]
    # [1, 2, -1, 3, -1, 0, 0]

    # [(4, -3), (2, 0), (3, 0), (1, 3)]
    # [(1, 3), (2, 0), (3, 0), (4, -3)]
    # [3, 0, 0, -3]

    import bisect

    def lis(a):
        dp = []
        for num in a:
            i = bisect.bisect_right(dp, num)
            if i == len(dp):
                dp.append(num)
            dp[i] = num
        return len(dp)

    def solve(n, a):
        b = [(x, i + 1 - x) for i, x in enumerate(a) if i + 1 - x >= 0]
        b.sort(key=lambda x: (x[0], -x[1]))
        b = list(map(lambda x: x[1], b))
        return lis(b)

    n = rn()
    a = rnl()
    print(solve(n, a))


import io
from alpha_codium.code_contests.eval.local_exec import swallow_io

inputs  = ['1\n2 0 0 0 0']
outputs = ['0']

# inputs= ['7\n'
# '2 1 4 2 5 3 7']
# outputs = ['3']
#
# # private tests:
# inputs = ['2 1\n-1000000000 1000000000\n2 1\n2 1 2\n',
#  '4 4\n2 -1000 100 3\n2 1\n3 2\n4 1\n2 1 3\n2 2 2\n1 1 -1000000000\n2 1 4\n']
# outputs = ['2000000000\n', '2102\n0\n1000000003\n']


# input_stream = io.BytesIO(single_input[0].encode())
input_stream = io.BytesIO(inputs[0].encode())
input_stream.seek(0)
try:
    with swallow_io(input_stream=input_stream) as (stdout_stream, stderr_stream):
        run()
    print(stdout_stream.getvalue())
except Exception as E:
    print(E)
    pass