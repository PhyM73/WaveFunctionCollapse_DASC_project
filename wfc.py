import math
import random
try:
    import Image
except:
    from PIL import Image


class ScanPattern:
    """ 
    扫描输入中的模式及其出现的频率，收集不同模式之间的邻近关系。
    
    模式(pattern)是进行分析的最小单元，大小为N*N。扫描输入矩阵，记录其中的模式并编号，将输入矩阵转化为由模式对应的编号组成的矩阵，同时记录模式间的邻近关系。如果允许对模式进行对称操作，将可能得到输入矩阵中不存在的模式。新模式之间的邻近关系同样由对称操作扩展。如果要求分析所有可能的邻近关系，则需要在各模式间进行邻近匹配。
     
    主要属性：
        entry: 输入矩阵
        patterns:{p1:i1, p2:i2, ...}        记录所有的模式
        revpatterns:{i1:p1, i2:p2, ...}     
        weights:[w1, w2, ...]     记录模式对应的频率(权重)
        matrix
        rules:[[{left}, {right}, {up}, {down}], [... ], ... ]        记录每个模式在各个方向上可以匹配的模式
    """

    def __init__(self, entry, N=3, symmetry=None, AllRules=False, LR_Reflect=False, UD_Reflect=False, Rotatable=False):
        self.width, self.height = len(entry) - N + 1, len(entry[0]) - N + 1
        self.N = N

        self.patterns, self.weights, self.rules = {}, [], []
        self.matrix = [[None] * self.height for _ in range(self.width)]
        index = 0
        for x in range(self.width):
            for y in range(self.height):
                pattern = self.get_pattern(entry, (x, y))
                try:
                    self.matrix[x][y] = self.patterns[pattern]
                    self.weights[self.matrix[x][y]] += 1
                except KeyError:
                    self.patterns[pattern] = self.matrix[x][y] = index
                    self.weights.append(1)
                    self.rules.append([set() for _ in range(4)])
                    index += 1
                self.make_rule((x, y))

        if symmetry != None:
            self.symmetry_operate(symmetry)
        self.revpatterns = {index: pattern for pattern, index in self.patterns.items()}

        if N > 1 and AllRules:
            self.make_allrule()  # 对所有pattern扫描匹配

        # self.changable = (LR_Reflect, UD_Reflect, Rotatable)  # (是否允许对模式做左右对称，上下对称，旋转) 准备删掉, 用symmetry来替代这部分的功能
        # self.count = len(self.patterns)

    def get_pattern(self, entry, position):
        '''获取以(x,y)为左上角的pattern'''
        x, y = position[0], position[1]
        N = self.N
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

    def symmetry_operate(self, symmetry):
        '''根据symmetry的要求对patterns和rules做对称操作'''  # 拟考虑按对称性组合分类来操作

        def horiz_reflect(pattern):  # 对pattern做水平反射
            return list(reversed(pattern))

        def vert_reflect(pattern):  # 对pattern做垂直反射
            return [list(reversed(pattern[i])) for i in range(len(pattern))]

        def diag_reflect(pattern):  # 对pattern做对角反射
            N = len(pattern)
            return [[pattern[y][x] for y in range(N)] for x in range(N)]

        def skew_reflect(pattern):  # 对pattern做反对角反射
            N = len(pattern)
            return [[pattern[N - y - 1][N - x - 1] for y in range(N)] for x in range(N)]

        def birotate(pattern):  # 旋转180°
            N = len(pattern)
            return [[pattern[N - x - 1][N - y - 1] for y in range(N)] for x in range(N)]

        def quadrotate(pattern):  # 旋转90°,180°,270°
            p = skew_reflect(pattern)
            return [p, horiz_reflect(p), vert_reflect(p)]

        symmetry_op = {
            'hr': horiz_reflect,
            'vr': vert_reflect,
            'dr': diag_reflect,
            'sr': skew_reflect,
            'bro': birotate,
            'qro': quadrotate
        }

        prime_patterns = self.patterns.copy()
        index = len(prime_patterns)
        for pattern, ind in prime_patterns:
            for sym in symmetry:
                p = symmetry_op[sym](pattern)
                try:
                    self.weights[prime_patterns[p]] *= 2
                except KeyError:
                    self.patterns[p] = index
                    self.weights.append(self.weights[ind])
                    # self.rules.append([set(None) for _ in range(4)])
                    # 需要根据对称性操作补上相应的规则
                    index += 1

    def make_allrule(self):
        '''遍历patterns匹配所有规则'''

        def overlap(pattern1, pattern2, direction):
            if direction == 0:
                return [line[:-1] for line in pattern1] == [line[1:] for line in pattern2]
            if direction == 2:
                return pattern1[:-1] == pattern2[1:]

        for index in range(len(self.patterns)):
            for ind in range(index):
                for direction in (0, 2):
                    if overlap(self.revpatterns[index], self.revpatterns[ind], direction):
                        self.rules[index][direction].add(ind)
                        self.rules[ind][direction + 1].add(index)


entry = [
    ['L', 'L', 'L', 'L'],
    ['L', 'L', 'L', 'L'],
    ['L', 'L', 'L', 'L'],
    ['L', 'C', 'C', 'L'],
    ['C', 'S', 'S', 'C'],
    ['S', 'S', 'S', 'S'],
    ['S', 'S', 'S', 'S'],
]
s = ScanPattern(list(entry))
print(s.width, s.height, s.weights)


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


class Lattice():
    '''格点单元'''

    def __init__(self, state_space):
        # 格点至少具备两个属性：状态空间，熵
        self.space = state_space  # state_space={state:wight}
        self.entropy = Lattice.shannon(state_space)

    @staticmethod
    def shannon(statespace):
        '''计算态空间的香农熵'''
        if len(statespace) == 1:
            return 0

        ws = sum(statespace.value())
        return math.log(ws) - sum(map(lambda x: x * math.log(x), statespace.value())) / ws


class Wave():
    '''体系波函数'''

    def __init__(self, size, state_space):
        # 波函数为包含所有格点的矩阵
        self.width, self.height = size[0], size[1]
        self.wave = [[Lattice(state_space)] * size[0]] * size[1]
        self.wait = []
        for i in range(size[0]):
            for j in range(size[1]):
                self.wait.append((i, j))

    def __getitem__(self, index):
        return self.wave[index[0]][index[1]]

    def min_entropy_pos(self):
        '''寻找熵最小的格点的位置'''
        x, y = self.wait[0][0], self.wait[0][1]
        min_entropy = self.wave[x][y].entropy
        for lattice in self.wait:
            if self.wave[lattice[0]][lattice[1]].entropy == 0:
                continue
            noise = random.random()
            if self.wave[lattice[0]][lattice[1]].entropy - noise < min_entropy:
                x, y = lattice[0], lattice[1]
                min_entropy = self.wave[x][y].entropy - noise

        # for i in range(self.width):
        # for j in range(self.height):
        # # if self.wave[x][y].entropy == 0:
        #     # continue
        # noise = random.random()
        # if self.wave[i][j].entropy - noise < min_entropy:
        #     x, y = i, j
        #     min_entropy = self.wave[x][y].entropy - noise
        return x, y

    def collapse(self):
        '''选择目前熵最小的格点，并随机塌缩'''
        x, y = self.min_entropy_pos()
        if len(self.wave[x][y].space) == 1:
            self.wave[x][y].entropy == 0
        elif len(self[x, y].space) > 1:
            s = random.choices(self[x, y].space.keys(), weights=self[x, y].space.value())
            self[x, y].space = {s, self[x, y].space[s]}
            del self.wait[x * self.width + y]
            self.propagate((x, y))
        else:
            print('Restart')
            pass

    def propagate(self, position):
        '''从给定位置处向周围传播塌缩'''
        pass

    def observe(self):
        '''测量整个波函数'''
        while self.isnot_all_collapsed():
            self.collapse()

    def isnot_all_collapsed(self):
        for x in self.width:
            for y in self.height:
                if self[x, y].entropy != 0:
                    return True
        return False