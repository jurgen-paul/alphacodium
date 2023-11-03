import io
import sys
from collections import defaultdict
from heapq import *

from alpha_codium.code_contests.eval.local_exec import swallow_io

def run():
    def max_f(a, n):
        dp = [0] * (n + 1)
        for i in range(n):
            dp[a[i]] = max(dp[a[i]], dp[a[i] - 1] + 1)
        return max(dp)

    if __name__ == "__main__":
        n = int(input().strip())
        a = list(map(int, input().strip().split()))
        print(max_f(a, n))
# # public test
# inputs  = ['1\n20000']
inputs  = ['5\n2 0 0 0 0']
outputs = ['0']
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