import io
import sys
from collections import defaultdict
from heapq import *

from alpha_codium.code_contests.eval.local_exec import swallow_io

def run():
    def initialize_dp(grid, n, m):
        dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        poles = []
        for i in range(n + 1):
            for j in range(m + 1):
                if grid[i][j] == 1:
                    dp[i][j] = 0
                    poles.append((i, j))
        return dp, poles

    def calculate_distances(dp, poles, n, m):
        for i in range(n + 1):
            for j in range(m + 1):
                for pole in poles:
                    dp[i][j] = min(dp[i][j], (i - pole[0]) ** 2 + (j - pole[1]) ** 2)
        return dp

    def sum_squares(dp, n, m):
        total = 0
        for i in range(n + 1):
            for j in range(m + 1):
                total += dp[i][j]
        return total

    def solve_problem():
        n, m = map(int, input().split())
        grid = [list(map(int, input())) for _ in range(n + 1)]
        dp, poles = initialize_dp(grid, n, m)
        dp = calculate_distances(dp, poles, n, m)
        total = sum_squares(dp, n, m)
        print(total)

    if __name__ == "__main__":
        solve_problem()
## private test

inputs  = ["""\
9 9
0000110000
0001001000
0001001000
0001111000
0001001000
0001001000
0110000110
1000000001
1001001001
0111111110\
"""]
outputs = ['182']


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