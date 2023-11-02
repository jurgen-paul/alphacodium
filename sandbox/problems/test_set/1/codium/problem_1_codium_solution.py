import io
import sys
from collections import defaultdict
from heapq import *

from alpha_codium.code_contests.eval.local_exec import swallow_io

def run():
    import math

    def calculate_distance(x, y):
        return math.sqrt(x ** 2 + y ** 2)

    def count_bird_habitats(distances, radius):
        return sum(1 for distance in distances if distance <= radius)

    def find_minimum_radius(n, k, bird_habitats):
        distances = sorted(calculate_distance(x, y) for x, y in bird_habitats)
        left, right = 0, distances[-1]
        while right - left > 1e-7:
            mid = (left + right) / 2
            if count_bird_habitats(distances, mid) >= k:
                right = mid
            else:
                left = mid
        return right

    if __name__ == "__main__":
        n, k = map(int, input().split())
        bird_habitats = [tuple(map(int, input().split())) for _ in range(n)]
        print("{:.10f}".format(find_minimum_radius(n, k, bird_habitats)))

# # public test
# single_input  = ['6 4\n10 -9 2 -1 4 -6\n1 5\n5 4\n5 6\n6 2\n6 3\n2 1 2\n1 1 -3\n2 1 2\n2 3 3\n']
# single_output = ['39\n32\n0\n']
#
# # private tests:
# inputs = ['2 1\n-1000000000 1000000000\n2 1\n2 1 2\n',
#  '4 4\n2 -1000 100 3\n2 1\n3 2\n4 1\n2 1 3\n2 2 2\n1 1 -1000000000\n2 1 4\n']
# outputs = ['2000000000\n', '2102\n0\n1000000003\n']


# input_stream = io.BytesIO(single_input[0].encode())
input_stream = io.BytesIO(inputs[1].encode())
input_stream.seek(0)
try:
    with swallow_io(input_stream=input_stream) as (stdout_stream, stderr_stream):
        run()
    print(stdout_stream.getvalue())
except Exception as E:
    print(E)
    pass