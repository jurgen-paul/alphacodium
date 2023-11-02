import io
from alpha_codium.code_contests.eval.local_exec import swallow_io

def run():
    def count_possible_numbers(s):
        n = len(s)
        if n == 1 and s == '0':
            return 1

        dp = [[[[0] * 2 for _ in range(10)] for _ in range(10)] for _ in range(n + 1)]
        dp[0][0][0][1] = 1

        for i in range(n):
            for j in range(10):
                for k in range(10):
                    for l in range(2):
                        for d in range(10):
                            if s[i] == '_' or s[i] == str(d) or (
                                    s[i] == 'X' and (i == 0 or s[i - 1] == 'X' or s[i - 1] == str(d))):
                                if l == 1 and d == 0 and (j != 0 or k != 0):
                                    continue
                                dp[i + 1][(k * 10 + d) % 10][d][l and d == 0] += dp[i][j][k][l]

        return sum(dp[n][2][5][l] + dp[n][7][5][l] for l in range(2))

    if __name__ == "__main__":
        s = input().strip()
        print(count_possible_numbers(s))
# # public test
inputs  = ['_XX']
outputs = ['9']
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