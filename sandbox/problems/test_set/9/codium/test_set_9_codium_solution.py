def read_input():
    n, m, k = map(int, input().split())
    grid = [list(map(int, input().split())) for _ in range(n)]
    balls = list(map(int, input().split()))
    return n, m, k, grid, balls

def drop_ball(n, m, grid, last_row, col):
    row = last_row[col]
    while row < n:
        direction = grid[row][col]
        if direction == 1:
            grid[row][col] = 2
            col += 1
        elif direction == 3:
            grid[row][col] = 2
            col -= 1
        else:
            row += 1
    last_row[col] = row
    return col

def solve(n, m, k, grid, balls):
    last_row = [0] * m
    result = []
    for ball in balls:
        result.append(drop_ball(n, m, grid, last_row, ball - 1) + 1)
    return result

def print_output(result):
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    n, m, k, grid, balls = read_input()
    result = solve(n, m, k, grid, balls)
    print_output(result)