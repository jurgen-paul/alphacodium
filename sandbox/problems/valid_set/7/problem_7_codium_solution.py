import io
import sys
from collections import defaultdict
from heapq import *

from alpha_codium.code_contests.eval.local_exec import swallow_io

def run():
    import sys
    from collections import defaultdict

    def read_input():
        t = int(input().strip())
        test_cases = []
        for _ in range(t):
            n = int(input().strip())
            a = list(map(int, input().strip().split()))
            test_cases.append((n, a))
        return t, test_cases

    def find_largest_friend_group(n, a):
        a.sort()
        max_a = a[-1]
        count = [0] * (max_a + 1)
        for i in a:
            count[i] += 1
        max_group_size = max(count)
        for i in range(2, max_a + 1):
            j = i
            temp = 0
            while j <= max_a:
                temp += count[j]
                j += i
            max_group_size = max(max_group_size, temp)
        return max_group_size

    if __name__ == "__main__":
        t, test_cases = read_input()
        for n, a in test_cases:
            print(find_largest_friend_group(n, a))
## private test

inputs  = ['1\n3\n1 137438953474 274877906947\n',
 '5\n1\n13\n1\n1\n6\n7 19 28 25 4 10\n3\n3 19 10\n9\n26 1 28 8 24 27 12 16 17\n',
 '1\n69\n2 4 10 16 22 28 34 40 46 52 58 64 70 76 82 88 94 100 106 112 118 124 130 136 142 148 154 160 166 172 178 184 190 196 202 208 214 220 226 232 238 244 250 256 262 268 274 280 286 292 298 304 310 316 322 328 334 340 346 352 358 364 370 376 382 388 391 397 403\n',
 '1\n20\n1 3 5 7 9 11 13 15 17 19 22 25 28 31 34 37 40 43 46 49\n',
 '1\n20\n16 15 17 8 30 23 20 28 27 6 1 18 24 2 10 5 14 29 12 7\n',
 '3\n3\n14 4 27\n13\n26 14 6 10 5 3 29 2 24 12 22 11 1\n4\n10 25 29 11\n',
 '1\n4\n1 140688977775057788 281377955550115575 281377955925200962\n']
outputs = ['3\n', '1\n1\n6\n2\n3\n', '68\n', '11\n', '4\n', '2\n4\n3\n', '4\n']


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