import io
import sys
from collections import defaultdict
from heapq import *

from alpha_codium.code_contests.eval.local_exec import swallow_io

def run():
    def read_input():
        n, q = map(int, input().split())
        a = list(map(int, input().split()))
        edges = [list(map(int, input().split())) for _ in range(n-1)]
        queries = [list(map(int, input().split())) for _ in range(q)]
        return n, q, a, edges, queries

    def build_graph(n, edges):
        graph = defaultdict(list)
        for u, v in edges:
            u -= 1
            v -= 1
            graph[u].append(v)
            graph[v].append(u)
        return graph

    def dfs(graph, parent, depth, u, p):
        parent[u] = p
        depth[u] = depth[p] + 1 if p != -1 else 0
        for v in graph[u]:
            if v == p:
                continue
            dfs(graph, parent, depth, v, u)

    def solve_query(a, parent, depth, u, v):
        lca = find_lca(parent, depth, u, v)
        return calculate_energy(a, parent, u, lca) + calculate_energy(a, parent, v, lca)

    def find_lca(parent, depth, u, v):
        if depth[u] < depth[v]:
            u, v = v, u
        while depth[u] > depth[v]:
            u = parent[u]
        while u != v:
            u = parent[u]
            v = parent[v]
        return u

    def calculate_energy(a, parent, u, lca):
        energy = 0
        while u != lca:
            energy += max(abs(a[u] + a[parent[u]]), abs(a[u] - a[parent[u]]))
            u = parent[u]
        return energy

    if __name__ == "__main__":
        n, q, a, edges, queries = read_input()
        graph = build_graph(n, edges)
        parent = [-1] * n
        depth = [0] * n
        dfs(graph, parent, depth, 0, -1)
        for query in queries:
            if query[0] == 1:
                a[query[1]-1] = query[2]
            else:
                print(solve_query(a, parent, depth, query[1]-1, query[2]-1))


# public test
single_input  = ['6 4\n10 -9 2 -1 4 -6\n1 5\n5 4\n5 6\n6 2\n6 3\n2 1 2\n1 1 -3\n2 1 2\n2 3 3\n']
single_output = ['39\n32\n0\n']

# private tests:
inputs = ['2 1\n-1000000000 1000000000\n2 1\n2 1 2\n',
 '4 4\n2 -1000 100 3\n2 1\n3 2\n4 1\n2 1 3\n2 2 2\n1 1 -1000000000\n2 1 4\n']
outputs = ['2000000000\n', '2102\n0\n1000000003\n']


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