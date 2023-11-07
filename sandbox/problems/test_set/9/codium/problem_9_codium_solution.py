import io

from alpha_codium.code_contests.eval.local_exec import swallow_io


def run():
    def calculate_color_combinations(n, m, k, r, c, a_x, a_y, b_x, b_y):
        MOD = 10**9 + 7
        total_cells = n * m
        sub_rectangle_cells = r * c
        unique_cells = total_cells - sub_rectangle_cells
        return pow(k, unique_cells, MOD)

    if __name__ == "__main__":
        n, m, k, r, c = map(int, input().split())
        a_x, a_y, b_x, b_y = map(int, input().split())
        print(calculate_color_combinations(n, m, k, r, c, a_x, a_y, b_x, b_y))


# private tests:
inputs = ['74 46 616259587 58 26\n'
'1 7 11 9']
outputs = ['894317354']


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