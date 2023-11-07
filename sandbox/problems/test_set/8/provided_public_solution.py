import io

from alpha_codium.code_contests.eval.local_exec import swallow_io


def run():
    import io, os

    # input = io.BytesIO(os.read(0, os.fstat(0).st_size)).readline
    from collections import deque

    INF = float('inf')


    class lca_binarylift(object):
        def __init__(self, neigh):
            set_root = 0  # the root of the tree
            self.n = len(neigh)
            self.parents = [[] for i in range(self.n)]
            self.depth = [-1] * (self.n)
            self.depth[set_root] = 0
            self.parents[set_root].append(-1)

            queue = deque([[set_root, 0]])
            while queue:
                index, d = queue.popleft()
                for nextindex in neigh[index]:
                    if self.depth[nextindex] >= 0:  continue
                    self.depth[nextindex] = d + 1
                    self.parents[nextindex].append(index)
                    queue.append([nextindex, d + 1])
            self.maxdepth = max(self.depth)

            k = 1
            while True:
                op = 0
                for i in range(self.n):
                    if len(self.parents[i]) == k and self.parents[i][k - 1] >= 0:
                        nextl = len(self.parents[self.parents[i][k - 1]])
                        actual = min(nextl - 1, k - 1)
                        self.parents[i].append(self.parents[self.parents[i][k - 1]][actual])
                        op += 1
                if op == 0: break
                k += 1

        def move(self, index, step):
            if step > self.depth[index]:  return -1
            i = 0
            while step:
                if step & 1:  index = self.parents[index][i]
                step = step >> 1
                i += 1
            return index

        def query(self, index1, index2):
            if self.depth[index1] >= self.depth[index2]:
                index1 = self.move(index1, self.depth[index1] - self.depth[index2])
            else:
                index2 = self.move(index2, self.depth[index2] - self.depth[index1])
            front = 0
            rear = self.maxdepth + 1
            while front < rear:
                mid = (front + rear) // 2
                if self.move(index1, mid) == self.move(index2, mid):
                    rear = mid
                else:
                    front = mid + 1
            return self.move(index1, front)


    class fenwick(object):
        def __init__(self, n):
            self.n = n
            self.cul = [0] * n

        def update(self, index, diff):
            i = index
            while i < self.n:
                self.cul[i] += diff
                i += (i + 1) & (-i - 1)

        def getaccu(self, index):
            output = 0
            i = index
            while i >= 0:
                output += self.cul[i]
                i -= (i + 1) & (-i - 1)
            return output

        def query(self, front, rear):
            return self.getaccu(rear) - self.getaccu(front - 1)


    class heavy_light(object):
        def __init__(self, n, neigh):
            self.n = n
            self.children = [[] for i in range(n)]
            self.neigh = neigh
            self.parent = [-1] * n
            self.ancestor = [-1] * n
            self.rename = [-1] * n
            self.totnum = [0] * n
            self.maxchild = [-1] * n
            self.renameindex()

        def getchild(self):
            visited = [False] * self.n
            queue = deque()
            queue.append(0)
            visited[0] = True
            seq = [0]
            while queue:
                index = queue.popleft()
                for nextindex in self.neigh[index]:
                    if visited[nextindex]: continue
                    visited[nextindex] = True
                    queue.append(nextindex)
                    self.children[index].append(nextindex)
                    self.parent[nextindex] = index
                    seq.append(nextindex)
            for index in seq[::-1]:

                maxcnum = 0
                for ele in self.children[index]:
                    self.totnum[index] += self.totnum[ele]
                    if self.totnum[ele] > maxcnum:
                        maxcnum = self.totnum[ele]
                        self.maxchild[index] = ele
                self.totnum[index] += 1

        def renameindex(self):
            self.getchild()
            stack = [(0, 0)]
            currindex = 0
            while stack:
                (index, ances) = stack.pop()
                for ele in self.children[index]:
                    if ele == self.maxchild[index]: continue
                    stack.append((ele, ele))
                self.ancestor[index] = ances
                self.rename[index] = currindex
                if self.maxchild[index] > 0:  stack.append((self.maxchild[index], ances))
                currindex += 1

        def getpath(self, index):
            output = []
            ori = index
            while index >= 0:
                front = self.rename[self.ancestor[index]]
                rear = self.rename[index]
                output.append([front, rear])
                index = self.parent[self.ancestor[index]]

            return output[::-1]


    n, q = map(int, input().split())
    arr = list(map(int, input().split()))
    for i in range(n):
        arr[i] = abs(arr[i])
    neigh = [[] for i in range(n)]
    for _ in range(n - 1):
        u, v = map(int, input().split())
        neigh[u - 1].append(v - 1)
        neigh[v - 1].append(u - 1)
    print(f"n:{n}, q:{q},neigh:{neigh}")
    # print(neigh)
    new = heavy_light(n, neigh)
    lca = lca_binarylift(neigh)
    # print(new.rename)
    fen = fenwick(n)
    for i in range(n):
        index = new.rename[i]
        fen.update(index, arr[i])
    for _ in range(q):
        op, a, b = map(int, input().split())

        if op == 1:
            i = a - 1
            index = new.rename[i]
            diff = abs(b) - arr[i]
            arr[i] = abs(b)
            fen.update(index, diff)
        else:
            front, rear = a - 1, b - 1
            oricommon = lca.query(a - 1, b - 1)
            to_rear = new.getpath(b - 1)
            to_front = new.getpath(a - 1)
            to_common = new.getpath(oricommon)
            #        print(front,rear,oricommon)
            #        print(to_rear,to_front,to_common)
            output = 0
            for ele in to_rear:
                output += fen.query(ele[0], ele[1])
            for ele in to_front:
                output += fen.query(ele[0], ele[1])
            for ele in to_common:
                output -= 2 * fen.query(ele[0], ele[1])
            output += arr[oricommon]
            output = 2 * output - arr[front] - arr[rear]

            print(output)



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