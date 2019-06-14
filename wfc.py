import math
import random
try:
    import Image
except:
    from PIL import Image
import numpy as np

#import matplotlib.pyplot as plt


def image2matrix(image_path):
    """Convert image at `image_path` to matrix."""
    image = Image.open(image_path)
    size = image.size
    load = image.load()
    return [[load[x, y] for y in range(size[1])] for x in range(size[0])]


class Grid():
    """Grid object which contains its state space and Shannon entropy."""

    def __init__(self, state_space):
        self.space = state_space  # state_space = {state:weight}
        self.entropy = Grid.shannon(state_space)

    @staticmethod
    def shannon(state_space):
        """Returns the Shannon entropy of the `state_space`."""
        if isinstance(state_space, int) or len(state_space) == 1:
            return 0
        ws = sum(state_space.values())
        return math.log(ws) - sum(map(lambda x: x * math.log(x), state_space.values())) / ws

    def __len__(self):
        return len(self.space)

    def __str__(self):
        return self.space


class CollapseError(StopIteration):
    pass


class WaveFunction():
    """Wave Function of the to-be-determined output
    
    首先扫描输入矩阵(entry)中的模式、其出现的频率和不同模式之间的邻近关系，进一步根据对称操作的要求补充模式和关系。模式(pattern)是进行分析的最小单元，大小为N*N。

    然后按照WFC算法对波函数进行观测。   

    Attributes：
        wave: A 2-D matrix which contains all the Grids.

        patterns: {i1:p1, i2:p2, ...}     
        weights: [w1, w2, ...]     
        rules: [ [{left}, {right}, {up}, {down}], [... ], ... ]        记录每个模式在各个方向上可以匹配的模式
     
    """

    def __init__(self, size, entry, N=3, AllRules=False, Periodic=False):
        # 初始化patterns
        self.patterns, self.weights, self.rules = {}, [], []
        self.BuildPatterns(entry, N=N, Periodic=Periodic)
        self.patterns = {index: pattern for pattern, index in self.patterns.items()}

        if N > 1 and AllRules:
            self.make_all_rules()

        self.image_size = size
        self.size = (size[0] - N + 1, size[1] - N + 1)
        self.wait_to_collapse = set((x, y) for x in range(self.size[0]) for y in range(self.size[1]))
        self.Stack = []  # 储存过程的栈，其中存储已经改变过的点坐标以及状态空间为元素为坐标到状态空间的字典

        state_space = {state: self.weights[state] for state in self.patterns.keys()}
        self.wave = [[Grid(state_space.copy()) for i in range(self.size[1])] for j in range(self.size[0])]
        # Initializes the 2-D WaveFunction matrix, each Grid of the matrix
        # starts with all states as possible. No state is forbidden yet.

        self.N = N
        self.image = self.buildimage()

    def BuildPatterns(self, entry, N=3, Periodic=False):
        """Parses the `entry` matrix. Extracts patterns, weights and adjacent rules. """
        if Periodic:
            width, height = len(entry) - 1, len(entry[0]) - 1
            entry = [entry[x][:] + entry[x][1:N - 1] for x in range(len(entry))]
            entry = entry[:] + entry[1:N - 1]
        else:
            width, height = len(entry) - N + 1, len(entry[0]) - N + 1
        matrix = [[None] * height for _ in range(width)]
        index = 0
        for x in range(width):
            for y in range(height):
                # Extract an N*N matrix as a pattern with the upper left corner being (x, y).
                pat = tuple(tuple(entry[x1][y:y + N]) for x1 in range(x, x + N))

                # If this pattern already exists, simply increment its weight.
                # Otherwise, records the new pattern and initializes its weight as 1, then increment the pattern index.
                try:
                    matrix[x][y] = self.patterns[pat]
                    self.weights[matrix[x][y]] += 1
                except KeyError:
                    self.patterns[pat] = matrix[x][y] = index
                    self.weights.append(1)
                    self.rules.append([set() for _ in range(4)])
                    index += 1
                self.make_rule((x, y), matrix, Periodic)

    def make_rule(self, position, matrix, Periodic):
        """为position处的pattern及其左侧、上侧的pattern创建邻近规则"""
        # The order of directions: (-1,0), (1,0), (0,-1), (0,1)
        (x, y) = position
        if x > 0:
            self.rules[matrix[x][y]][0].add(matrix[x - 1][y])
            self.rules[matrix[x - 1][y]][1].add(matrix[x][y])
        if y > 0:
            self.rules[matrix[x][y]][2].add(matrix[x][y - 1])
            self.rules[matrix[x][y - 1]][3].add(matrix[x][y])

    def make_all_rules(self):
        """Traverses patterns to match all the possible rules.
        This method may allow some adjacent rules that do not exist in the original input."""

        def compatible(pattern1, pattern2, direction):
            """Returns `True` if `pattern2` is compatible with `pattern1` in the `direction`,
            otherwise return `False`."""
            if direction == 0:
                return pattern1[:-1] == pattern2[1:]
            if direction == 2:
                return [line[:-1] for line in pattern1] == [line[1:] for line in pattern2]

        for index in range(len(self.patterns)):
            for ind in range(index + 1):
                for direction in (0, 2):
                    if compatible(self.patterns[index], self.patterns[ind], direction):
                        self.rules[index][direction].add(ind)
                        self.rules[ind][direction + 1].add(index)

    def buildimage(self):
        weights = np.array(self.weights)
        mean = tuple(
            map(lambda x: int(np.average(np.array(x), weights=weights)),
                zip(*(pattern[0][0] for pattern in self.patterns.values()))))
        return Image.new('RGB', self.image_size, mean)

    def update(self, position):
        image = self.image.load()
        limit_i, limit_j = 1, 1
        if position[0] == self.size[0] - 1:
            limit_i = self.N
        if position[1] == self.size[1] - 1:
            limit_j = self.N
        for i in range(limit_i):
            for j in range(limit_j):
                x, y = position[0] + i, position[1] + j
                keys, values = list(self[position].space.keys()), np.array(list(self[position].space.values()))
                #print(keys, values)
                mean = tuple(
                    map(lambda x: int(np.average(np.array(x), weights=values)),
                        zip(*(self.patterns[index][i][j] for index in keys))))
                image[x, y] = mean

    def __getitem__(self, index):
        return self.wave[index[0]][index[1]]

    def __setitem__(self, index, value):
        self.wave[index[0]][index[1]] = value

    def min_entropy_pos(self):
        """Returns the position of the Grid whose statespace has the lowest entropy."""
        min_entropy = float("inf")
        for lattice in self.wait_to_collapse:
            noise = random.random() / 1000
            # Add some noise to mix things up a little
            if self[lattice].entropy - noise < min_entropy:
                position = lattice[:]
                min_entropy = self[position].entropy - noise
        return position

    def neighbor(self, position):
        """Yields neighboring Grids and their directions of a given `position`."""
        if position[0] > 0:
            yield (position[0] - 1, position[1]), 0
        if position[0] < self.size[0] - 1:
            yield (position[0] + 1, position[1]), 1
        if position[1] > 0:
            yield (position[0], position[1] - 1), 2
        if position[1] < self.size[1] - 1:
            yield (position[0], position[1] + 1), 3

    def collapse(self, position):
        """Collapses the grid at `position`, and then propagates the consequences. """
        (x, y) = position
        if len(self[position]) < 1:
            self.backtrack()
        else:
            if len(self[position]) > 1:
                # Choose one possible pattern randomly and push this changed Grid into the Stack.
                states, w = list(self[position].space.keys()), list(self[position].space.values())
                elem = random.choices(states, weights=w)[0]
                del self.wave[x][y].space[elem]
                self.Stack.append({position: self[position].space.copy()})
                self[x, y] = Grid({elem: 1})
            self.update(position)
            self.wait_to_collapse.remove(position)
            self.propagate(position)

    def propagate(self, position):
        """Propagates the consequences of the wavefunction collapse or statespace changing at `position`.
        This method keeps propagating the consequences of the consequences,and so on until no consequences remain. 
        """
        PropagStack = [position]

        while PropagStack:
            pos = PropagStack.pop()

            for nb, direction in self.neighbor(pos):
                if nb in self.wait_to_collapse:
                    available = set.union(*[self.rules[state][direction] for state in self[pos].space.keys()])
                    if not set(self[nb].space.keys()).issubset(available):
                        available = available & set(self[nb].space.keys())
                        if len(available) == 0:
                            self.backtrack()
                            break
                        elif self.Stack and (nb not in self.Stack[-1].keys()):
                            # push this changed Grid into the Stack.
                            self.Stack[-1][nb] = self[nb].space.copy()
                        self[nb] = Grid({state: self.weights[state] for state in available})
                        self.update(nb)
                        PropagStack.append(nb)

    def backtrack(self):
        """Backtracks to the previous step. 
        If there is no way to backtrack then this method raises CollapseError. """
        print('0')
        if self.Stack:
            step = self.Stack.pop()
            # Restore all the Girds affected by the last collapse
            for (position, space) in step.items():
                self[position] = Grid(space)
                self.wait_to_collapse.add(position)
        else:
            raise CollapseError("No Sulotion")

    def observe(self):
        """Observe the whole WaveFunction"""
        while self.wait_to_collapse:
            # yield self.image
            self.collapse(self.min_entropy_pos())
        yield self.image


# entry = [
#     # ['S', 'S', 'S', 'C', 'L', 'L', 'L', 'L', 'L', 'L', 'L'],
#     ['S', 'S', 'C', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L'],
#     ['C', 'C', 'C', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L'],
#     ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L'],
#     ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'C', 'L', 'L'],
#     ['L', 'L', 'C', 'C', 'L', 'L', 'L', 'C', 'S', 'C', 'L'],
#     ['L', 'C', 'S', 'S', 'C', 'L', 'L', 'C', 'S', 'C', 'L'],
#     ['C', 'S', 'S', 'S', 'S', 'C', 'C', 'S', 'S', 'S', 'C'],
#     ['C', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C', 'L'],
#     ['C', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C', 'L', 'L'],
#     ['C', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C', 'L'],
#     ['C', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C'],
#     ['C', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C'],
#     ['L', 'C', 'S', 'S', 'S', 'C', 'S', 'S', 'S', 'C', 'L'],
#     ['L', 'L', 'C', 'C', 'C', 'L', 'C', 'S', 'C', 'L', 'L'],
#     ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'C', 'L', 'L', 'L'],
#     ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L'],
#     ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L'],
#     ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L'],
#     ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L'],
#     ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L'],
# ]
# ['C', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C', 'L'],
entry = image2matrix(r"samples\Colored City.png")  #路径前加r转义，r'*****'

# # 处理图片时调用
# image1 = Image.new('RGB', (70, 70), (0, 0, 0))
# result = image1.load()
# for i in range(w.size[0]):
#     for j in range(w.size[1]):
#         result[i, j] = w.patterns[list(w[i, j].space.keys())[0]][0][0]
# image1.save('emmmm.png')
# image1.show()
# 处理图片时调用

#image1 = Image.new('RGB', (40, 40), (0, 0, 0))
#result = image1.load()

for w in WaveFunction((60, 60), entry, N=1, Periodic=False).observe():
    w.save('e.bmp')
    w.show()

#处理矩阵时调用
# result = [[None] * w.size[1] for _ in range(w.size[0])]
# for i in range(w.size[0]):
#     for j in range(w.size[1]):
#         result[i][j] = w.patterns[list(w.wave[i][j].space.keys())[0]][0][0]
# for line in result:
#     for i in line:
#         print(i, end='')
#     print('')
