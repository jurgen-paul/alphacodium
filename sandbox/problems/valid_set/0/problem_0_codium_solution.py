import io
import sys
from collections import defaultdict
from heapq import *

from alpha_codium.code_contests.eval.local_exec import swallow_io

def run():
    def initialize_table(n):
        table = [[0 for _ in range(3 * n + 1)] for _ in range(n + 1)]
        table[0][0] = 1
        return table

    def fill_table(table, n):
        MOD = 10 ** 9 + 7
        for i in range(1, n + 1):
            for j in range(3 * n + 1):
                table[i][j] = table[i - 1][j]
                if j >= 1:
                    table[i][j] = (table[i][j] + table[i - 1][j - 1]) % MOD
                if j >= 2:
                    table[i][j] = (table[i][j] + table[i - 1][j - 2]) % MOD
                if j >= 3:
                    table[i][j] = (table[i][j] + table[i - 1][j - 3]) % MOD
        return table

    def calculate_attack_plans(table, n, queries):
        results = []
        for x in queries:
            result = 0
            for i in range(1, min(n, x // 3) + 1):
                result = (result + table[i][x]) % (10 ** 9 + 7)
            results.append(result)
        return results

    if __name__ == "__main__":
        n, q = map(int, input().split())
        queries = [int(input()) for _ in range(q)]
        table = initialize_table(n)
        table = fill_table(table, n)
        results = calculate_attack_plans(table, n, queries)
        for result in results:
            print(result)
## private test

inputs  = ["""\
2 3
1
5
6
"""]
outputs = ["""\
9
6
1
"""]


# input_stream = io.BytesIO(single_input[0].encode())
input_stream = io.BytesIO(inputs[4].encode())
input_stream.seek(0)
try:
    with swallow_io(input_stream=input_stream) as (stdout_stream, stderr_stream):
        run()
    print(stdout_stream.getvalue())
except Exception as E:
    print(E)
    pass