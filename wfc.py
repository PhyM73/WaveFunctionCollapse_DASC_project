import math
import random
try:
    import Image
except:
    from PIL import Image


def transimage(image_path):  #将图片转化为矩阵
    image = Image.open(image_path)
    wid, hei = image.size
    load = image.load()
    matrix = [[None] * hei for _ in range(wid)]
    for x in range(wid):
        for y in range(hei):
            matrix[x][y] = load[x, y]
    return matrix


class ScanPattern:
    """ 
    扫描输入中的模式及其出现的频率，收集不同模式之间的邻近关系。
    
    模式(pattern)是进行分析的最小单元，大小为N*N。扫描输入矩阵，记录其中的模式并编号，将输入矩阵转化为由模式对应的编号组成的矩阵，同时记录模式间的邻近关系。如果允许对模式进行对称操作，将可能得到输入矩阵中不存在的模式。新模式之间的邻近关系同样由对称操作扩展。如果要求分析所有可能的邻近关系，则需要在各模式间进行邻近匹配。
     
    主要属性：
        entry: 输入矩阵
        patterns:{p1:i1, p2:i2, ...}        记录所有的模式(后转为{i：p})    
        weights:[w1, w2, ...]     记录模式对应的频率(权重)
        matrix
        rules:[ [{left}, {right}, {up}, {down}], [... ], ... ]        记录每个模式在各个方向上可以匹配的模式
    """

    def __init__(self, entry, N=3, symmetry=None, Reflect=None, Rotate=None, AllRules=False):
        self.width, self.height = len(entry) - N + 1, len(entry[0]) - N + 1

        self.patterns, self.weights, self.rules = {}, [], []
        self.matrix = [[None] * self.height for _ in range(self.width)]
        index = 0
        for x in range(self.width):
            for y in range(self.height):
                pat = self.get_pattern(entry, (x, y), N=N)
                try:
                    self.matrix[x][y] = self.patterns[pat]
                    self.weights[self.matrix[x][y]] += 1
                except KeyError:
                    self.patterns[pat] = self.matrix[x][y] = index
                    self.weights.append(1)
                    self.rules.append([set() for _ in range(4)])
                    index += 1
                self.make_rule((x, y))
        self.prim = self.patterns.copy()

        # if Reflect != None or Rotate != None:
        #     self.symmetry_operate(Reflect,Rotate)
        # if symmetry != None:
        # self.symmetry_operate(symmetry)

        self.patterns = {index: pattern for pattern, index in self.patterns.items()}

        if N > 1 and AllRules:
            self.make_allrule()  # 对所有pattern扫描匹配

    def get_pattern(self, entry, position, N):
        '''获取以(x,y)为左上角的pattern'''
        x, y = position[0], position[1]
        return tuple(tuple(entry[x1][y:y + N]) for x1 in range(x, x + N))

    def make_rule(self, position):
        '''为(x,y)处的pattern及其左侧、上侧的pattern创建邻近规则'''
        # [{left}, {right}, {up}, {down}]
        x, y = position[0], position[1]
        if x > 0:
            self.rules[self.matrix[x][y]][0].add(self.matrix[x - 1][y])
            self.rules[self.matrix[x - 1][y]][1].add(self.matrix[x][y])
        if y > 0:
            self.rules[self.matrix[x][y]][2].add(self.matrix[x][y - 1])
            self.rules[self.matrix[x][y - 1]][3].add(self.matrix[x][y])

    # def symmetry_operate(self, Reflect, Rotate):
    #     '''根据Reflect和Rotate的要求对patterns和rules做对称操作'''  # 拟考虑按对称性组合分类来操作
    #     def horiz_reflect(pattern):  # 对pattern做水平反射
    #         return list(reversed(pattern))

    #     def vert_reflect(pattern):  # 对pattern做垂直反射
    #         return [list(reversed(pattern[i])) for i in range(len(pattern))]

    #     def diag_reflect(pattern):  # 对pattern做对角反射
    #         N = len(pattern)
    #         return [[pattern[y][x] for y in range(N)] for x in range(N)]

    #     def skew_reflect(pattern):  # 对pattern做反对角反射
    #         N = len(pattern)
    #         return [[pattern[N - y - 1][N - x - 1] for y in range(N)] for x in range(N)]

    #     def birotate(pattern):  # 旋转180°
    #         N = len(pattern)
    #         return [[pattern[N - x - 1][N - y - 1] for y in range(N)] for x in range(N)]

    #     def quadrotate(pattern):  # 旋转90°,180°,270°
    #         p = skew_reflect(pattern)
    #         return [p, horiz_reflect(p), vert_reflect(p)]

    #     symmetry_op = {
    #         'hr': horiz_reflect,
    #         'vr': vert_reflect,
    #         'dr': diag_reflect,
    #         'sr': skew_reflect,
    #         'bro': birotate,
    #         'qro': quadrotate
    #     }

    # def rotate(times, ref=None):
    #     if ref:
    #         pass
    #     elif times == 2:
    #         pass
    #     elif times == 4:
    #         pass
    #     else:
    #         raise ValueError

    # if Rotate = None:
    #     pass
    # elif Reflect = None:
    #     pass  # rotate(rotate)
    # else:
    #     if Rotate == 4:
    #         pass
    #     elif Rotate == 2:
    #         pass
    #     else:
    #         raise ValueError

    # prime_patterns = self.patterns.copy()
    # index = len(prime_patterns)
    # for pattern, ind in prime_patterns:
    #     # for sym in symmetry:
    #         p = symmetry_op[sym](pattern)
    #         try:
    #             self.weights[prime_patterns[p]] *= 2
    #         except KeyError:
    #             self.patterns[p] = index
    #             self.weights.append(self.weights[ind])
    #             # self.rules.append([set(None) for _ in range(4)])
    #             # 需要根据对称性操作补上相应的规则
    #             index += 1

    # def rotate(self, times, ref=None,N=3):
    #     index = len(self.patterns)
    #     if ref:
    #         pass
    #     elif times == 2:
    #         for pattern, ind in self.prim.items():
    #             p=[[pattern[N - x - 1][N - y - 1] for y in range(N)] for x in range(N)]
    #     elif times == 4:
    #         pass
    #     else:
    #         raise ValueError

    def make_allrule(self):
        '''遍历patterns匹配所有规则'''

        def overlap(pattern1, pattern2, direction):
            if direction == 0:
                return pattern1[:-1] == pattern2[1:]
            if direction == 2:
                return [line[:-1] for line in pattern1] == [line[1:] for line in pattern2]

        for index in range(len(self.patterns)):
            for ind in range(index + 1):
                for direction in (0, 2):
                    if overlap(self.patterns[index], self.patterns[ind], direction):
                        self.rules[index][direction].add(ind)
                        self.rules[ind][direction + 1].add(index)


class ScanTile:
    #可以看成ScanPattern类N=1的实现，可以作为一个尝试保留
    def __init__(self, image_path):
        self.image = Image.open(image_path)
        self.width = self.image.width
        self.height = self.image.height
        self.matrix = self.image2matrix()
        self.tiles = set()  #{tile1, tile2, tile3, ...}
        self.weights = dict()  #{tile1:weight, tile2:weight, ...}
        self.rules = dict(
        )  #self.rules -- {tile1:{left:{tile1:count, tile2:count,...}, right:{...}, up:{...}, down:{...}}, tile2:{...}, ...}

    def image2matrix(self):
        #将输入的图片转化为矩阵，矩阵元为像素值tuple
        #暂时还只能将每个像素当作一个tile，可以尝试将图片分隔为由多个像素点组成的网格，每个网格当作一个tile
        matrix = [[None] * self.height for _ in range(self.width)]
        for x in range(self.width):
            for y in range(self.height):
                matrix[x][y] = self.image.getpixel((x, y))
        return matrix

    def get_rules(self):
        '''之后调用此函数可得到输入矩阵的所有信息
        如:tiles, weights, rules = Scan(Matrix).get_rules()
        '''
        #构建self.tiles、self.weights
        for x in range(self.width):
            for y in range(self.height):
                if self.matrix[x][y] not in self.tiles:
                    self.tiles.add(self.matrix[x][y])
                    self.weights[self.matrix[x][y]] = 1
                else:
                    self.weights[self.matrix[x][y]] += 1
        #构建self.rules
        left = (-1, 0)
        right = (1, 0)
        up = (0, -1)
        down = (0, 1)
        directions = {left, right, up, down}
        rule_in_one_dir = dict.fromkeys(self.tiles, 0)

        for tile in self.tiles:
            self.rules[tile] = {
                left: rule_in_one_dir.copy(),
                right: rule_in_one_dir.copy(),
                up: rule_in_one_dir.copy(),
                down: rule_in_one_dir.copy()
            }
        for x in range(self.width):
            for y in range(self.height):
                for direction in directions:
                    x1 = x + direction[0]
                    y1 = y + direction[1]
                    if x1 >= 0 and x1 < self.width and y1 >= 0 and y1 < self.height:
                        self.rules[self.matrix[x][y]][direction][self.matrix[x1][y1]] += 1
        return self.tiles, self.weights, self.rules


# tiles, weights, rules = ScanTile(r'image_path').get_rules()


class Grid():
    ''''''

    def __init__(self, state_space):
        self.space = state_space  # state_space = {state:weight}

        def shannon(state_space):
            '''Return the shannon entropy of the state_space'''
            if len(state_space) == 1:
                return 0

            ws = sum(state_space.values())
            return math.log(ws) - sum(map(lambda x: x * math.log(x), state_space.values())) / ws

        self.entropy = shannon(state_space)

    def __len__(self):
        return len(self.space)

    def __str__(self):
        return self.space


class CollapseError(StopIteration):
    pass


class Wave():
    '''体系波函数
    
    主要属性：
        wave: 存储格点(grid)的矩阵
        rules: 承接输入的rules
    '''

    def __init__(self, size, charts):
        # 波函数为包含所有格点的矩阵
        self.width, self.height = size[0], size[1]
        self.wait_to_collapse = set((i, j) for i in range(size[0]) for j in range(size[1]))
        # for i in range(size[0]):
        # self.wait_to_collapse |= set((i, j) for j in range(size[1]))
        # for j in range(size[1]):
        # self.wait_to_collapse.add((i, j))

        self.Stack = []  #储存过程的栈，其中存储已经改变过的点坐标以及状态空间为元素为坐标到状态空间的字典
        self.weight = charts.weights

        state_space = {state: charts.weights[state] for state in charts.patterns.keys()}
        self.wave = [[Grid(state_space.copy()) for i in range(size[1])] for j in range(size[0])]
        self.rules = charts.rules

    def __getitem__(self, index):
        return self.wave[index[0]][index[1]]

    def min_entropy_pos(self):
        '''寻找熵最小的格点的位置'''
        min_entropy = float("inf")
        for lattice in self.wait_to_collapse:
            #if self[lattice[0], lattice[1]].entropy == 0:
            #continue
            #noise = random.random() / 1000
            if self[lattice].entropy < min_entropy:
                position = lattice[:]
                min_entropy = self[position].entropy
        return position

    def collapse(self, position):
        '''选择目前熵最小的格点，并随机塌缩'''
        x, y = position[0], position[1]
        if len(self[x, y]) < 1:
            #raise CollapseError
            self.goback()
        elif len(self[x, y]) == 1:  #没有另外的选择，不入栈
            self.wave[x][y].space = self[x, y].space.popitem()[0]
            self.wave[x][y].entropy = 0
            self.wait_to_collapse.remove(position)
            self.propagate(position)
        else:  #有另外的选择，在状态空间删去已选的pattern后入栈
            states, w = list(self[x, y].space.keys()), list(self[x, y].space.values())
            elem = random.choices(states, weights=w)[0]
            del self.wave[x][y].space[elem]
            self.Stack.append({(x, y): self.wave[x][y].space.copy()})
            self.wave[x][y].space = elem
            self.wave[x][y].entropy = 0
            self.wait_to_collapse.remove(position)
            self.propagate((x, y))

    def neighbor(self, position):
        if position[0] > 0:
            yield (position[0] - 1, position[1]), 0
        if position[0] < self.width - 1:
            yield (position[0] + 1, position[1]), 1
        if position[1] > 0:
            yield (position[0], position[1] - 1), 2
        if position[1] < self.height - 1:
            yield (position[0], position[1] + 1), 3

    def propagate(self, position):
        '''从给定位置处向周围传播塌缩'''
        for nb, direction in self.neighbor(position):
            if nb in self.wait_to_collapse:
                available = self.rules[self[position].space][direction] & set(self[nb].space.keys())
                if len(available) == 0:
                    self.goback()
                    break
                # if len(available) == 1:
                #     self.near.append(nb)
                if self.Stack and (nb not in self.Stack[-1].keys()):
                    self.Stack[-1][nb] = self[nb].space.copy()  #加如到引起此变化的塌缩点所在的字典中，并且只记录最初的状态空间
                self.wave[nb[0]][nb[1]] = Grid({state: self.weight[state] for state in available})
            else:
                if self[nb].space in self.rules[self[position].space][direction]:
                    ''' 按逻辑来说，如果nb点已塌缩，那么position点的状态空间已经收到nb塌缩的影响，因此这个判断应该恒为True，但是在处理图片以及较大矩阵
                    (大概为30*30)时有时会为False(处理图片时几乎每次都出现)，具体原因还没找到 '''
                    #如果去掉此判断，可以得到一些图片，但是其中有不符合规则的地方
                    continue
                #找bug时加的部分
                # else:
                #     result = [[None] * self.height for _ in range(self.width)]
                #     for i in range(self.width):
                #         for j in range(self.height):
                #             if isinstance(self[i,j].space, int):
                #                 result[i][j] = self[i,j].space
                #             else:
                #                 result[i][j] = ' '
                #     result[nb[0]][nb[1]] += 1000
                #     result[position[0]][position[1]] += 100
                #     for line in result:
                #         print(line)
                #     print('wrong')
                #     print(s.patterns[self[nb].space])
                #     print(s.patterns[self[position].space])
                #     for i in range(5):
                #         print(self.Stack.pop())
                #     raise CollapseError('wrong')

    def goback(self):  #回溯
        if self.Stack:
            step = self.Stack.pop()
            for ((x, y), space) in step.items():  #将最后一次塌缩影响过的所有点还原
                self.wave[x][y] = Grid(space)
                self.wait_to_collapse.add((x, y))
        else:
            raise CollapseError("No Sulotion")

    def observe(self):
        '''测量整个波函数'''
        while self.wait_to_collapse:
            self.collapse(self.min_entropy_pos())
        return self.wave

    # def waveprint(self):
    #     for x in range(self.width):
    #         print('')
    #         for y in range(self.height):
    #             if isinstance(self.wave[x][y].space, dict):
    #                 print(['\\'], end='')
    #             else:
    #                 print(self.wave[x][y].space, end='')


entry = [
    ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L'],
    ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L'],
    ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L'],
    ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L'],
    ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L'],
    ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L'],
    ['L', 'L', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'L', 'L'],
    ['L', 'C', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C', 'L'],
    ['C', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C'],
    ['C', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C'],
    ['C', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C'],
    ['C', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C'],
    ['C', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C'],
    ['C', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C'],
    ['C', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C'],
    ['L', 'C', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C', 'L'],
    ['L', 'L', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'L', 'L'],
    ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L'],
    ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L'],
    ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L'],
    ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L'],
    ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L'],
]
#entry = transimage(image_path)#路径前加r转义，r'*****'
s = ScanPattern(entry, N=2)

# print(s.width, s.height, s.weights)
# print(s.patterns)
# print(s.rules)
""" def wfc(entry, max_iter=1000):
    i = 0
    while i < max_iter:
        try:
            w = Wave((8, 8), entry).observe()
            # print(w)
            break
        except CollapseError:
            if i == max_iter - 1:
                raise ValueError
            i += 1
            # print('fail')
    return w """

w = Wave((30, 30), s).observe()

#处理图片时调用
# image1 = Image.new('RGB', (30,30), (0,0,0))
# result = image1.load()
# for i in range(len(w)):
#     for j in range(len(w[0])):
#         result[i,j] = s.patterns[w[i][j].space][0][0]
# image1.save('emmmm.png')
# image1.show()

#处理矩阵时调用
result = [[None] * len(w[0]) for _ in range(len(w))]
for i in range(len(w)):
    for j in range(len(w[0])):
        result[i][j] = w[i][j].space
for line in result:
    print(line)
