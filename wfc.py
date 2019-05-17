import math
import random
try:
    import Image
except:
    from PIL import Image

# class ScanPattern:
#     """
#     扫描输入的图片，读取图中的模式及其出现的频率，收集不同模式之间的邻近关系。

#     将输入的图片转化为由 N*N 的模式(pattern)组成的矩阵，通过遍历的方式记录各模式的出现频率，如果允许对称、旋转处理，对所有模式做变换得到更多模式。

#     主要属性：
#         matrix:输入图像转为pattern
#         patterns:{p1:i1, p2:i2, ...}        记录所有的模式
#         revpatterns:{i1:p1, i2:p2, ...}
#         weights:[w1, w2, ...]     记录模式对应的频率(权重)
#         rules:[[{p1, p2, ...}, { set }, ], [... ], ... ]        记录每个模式在各个方向上可以匹配的模式
#     """

#     def __init__(self, image_path, N=3, LR_Reflect=False, UD_Reflect=False, Rotatable=False):
#         self.image = Image.open(image_path)  #打开图片
#         self.N = N  #模式的大小
#         self.changable = (LR_Reflect, UD_Reflect, Rotatable)  #(是否允许对模式做左右对称，上下对称，旋转)
#         self.width = self.image.width - N + 1
#         self.height = self.image.height - N + 1

#         self.build()

#         # self.matrix, self.patterns, self.weights = self.build()
#         # self.matrix = self.image2matrix()  #由pattern构成的矩阵
#         # self.patterns, self.weights = self.build_patterns()
#         # self.num_matrix = self.build_num_matrix()  #将pattern用它在patterns中的index表示
#         # self.count = len(self.patterns)  #pattern总数
#         self.rules = self.get_rules()

#     # def image2matrix(self):
#     #     matrix = [[None] * self.height for _ in range(self.width)]
#     #     for x in range(self.width):
#     #         for y in range(self.height):
#     #             matrix[x][y] = self.get_pattern((x, y))
#     #     return matrix

#     def get_pattern(self, position):
#         # 以(x,y)为左上顶点构建pattern
#         x, y = position[0], position[1]
#         N = self.N
#         pattern = [[None] * N for _ in range(N)]
#         for x1 in range(N):
#             for y1 in range(N):
#                 pattern[x1][y1] = self.image.getpixel((x + x1, y + y1))
#         return pattern

#     # def build_num_matrix(self):
#     #     num_matrix = [[None] * self.height for _ in range(self.width)]
#     #     for x in range(self.width):
#     #         for y in range(self.height):
#     #             num_matrix[x][y] = self.patterns.index(self.matrix[x][y])
#     #     return num_matrix

#     def build_patterns(self):
#         '''构建并返回patterns和weights'''

#         def LR_Reflect(pattern):  #对pattern做左右对称
#             return list(reversed(pattern))

#         def UD_Reflect(pattern):  #对pattern做上下对称
#             return [list(reversed(pattern[i])) for i in range(N)]

#         def Rotated(pattern):
#             #对pattern做旋转、转置，返回一个由转置、旋转90°、270°、转置并旋转180°后的pattern组成的list
#             def transposition(pattern):  #将一个pattern转置
#                 temp = [[None] * N for _ in range(N)]
#                 for x in range(N):
#                     for y in range(N):
#                         temp[y][x] = pattern[x][y]
#                 return temp

#             p1 = transposition(pattern)
#             p2 = LR_Reflect(p1)
#             p3 = UD_Reflect(p1)
#             p4 = UD_Reflect(p2)
#             return [p1, p2, p3, p4]

#         def extend(patterns, weights, changable):
#             # 如果允许，基于图中提取的pattern通过对称、旋转、转置等变换构造更多的pattern,构造出的pattern拥有与它们的来源相同的weight
#             temp_patterns, temp_weights = patterns[:], weights
#             weights = dict.fromkeys(range(len(patterns)), 0)
#             patterns = patterns
#             for pattern in temp_patterns:
#                 current_patterns = [pattern]  #current_patterns = [当前pattern，左右对称，上下左右对称，上下对称，转置，旋转90°，旋转270°，转置并旋转180°]
#                 if changable[0]:
#                     current_patterns.append(LR_Reflect(current_patterns[0]))
#                 if changable[1] and changable[0]:
#                     current_patterns.append(UD_Reflect(current_patterns[1]))
#                 if changable[1]:
#                     current_patterns.append(UD_Reflect(current_patterns[0]))
#                 if changable[2]:
#                     current_patterns.extend(Rotated(current_patterns[0]))
#                 for each in current_patterns:
#                     if each not in patterns:
#                         patterns.append(each)
#                         weights[patterns.index(each)] = temp_weights[temp_patterns.index(pattern)]
#                     else:
#                         weights[patterns.index(each)] += temp_weights[temp_patterns.index(pattern)]
#             return patterns, weights

#         patterns = []
#         weights = dict()
#         changable = self.changable
#         N = self.N
#         # 基于样图构造patterns与weights
#         for x in range(self.width):
#             for y in range(self.height):
#                 pattern = self.matrix[x][y]
#                 if pattern not in patterns:
#                     patterns.append(pattern)
#                     weights[patterns.index(pattern)] = 1
#                 else:
#                     weights[patterns.index(pattern)] += 1
#         if sum(changable):
#             # 允许变换，对patterns进行拓展
#             patterns, weights = extend(patterns, weights, changable)
#         return patterns, weights

#     def build(self):
#         '''将输入的图像转为矩阵，并构建patterns和weights'''
#         patterns = {}
#         weights = []
#         matrix = [[None] * self.height for _ in range(self.width)]
#         N = self.N
#         index = 0
#         for x in range(self.width):
#             for y in range(self.height):
#                 pattern = self.get_pattern((x, y))
#                 try:
#                     i = patterns[pattern]
#                     matrix[x][y] = i
#                     weights[i] += 1
#                 except KeyError:
#                     patterns[pattern] = index
#                     matrix[x][y] = index
#                     weights.append(1)
#                     index += 1

#         def LR_Reflect(pattern):  #对pattern做左右对称
#             return list(reversed(pattern))

#         def UD_Reflect(pattern):  #对pattern做上下对称
#             return [list(reversed(pattern[i])) for i in range(N)]

#         def Rotated(pattern):
#             #对pattern做旋转、转置，返回一个由转置、旋转90°、270°、转置并旋转180°后的pattern组成的list
#             def transposition(pattern):  #将一个pattern转置
#                 temp = [[None] * N for _ in range(N)]
#                 for x in range(N):
#                     for y in range(N):
#                         temp[y][x] = pattern[x][y]
#                 return temp

#             p1 = transposition(pattern)
#             p2 = LR_Reflect(p1)
#             p3 = UD_Reflect(p1)
#             p4 = UD_Reflect(p2)
#             return [p1, p2, p3, p4]

#         def extend(patterns, weights, changable):
#             # 如果允许，基于图中提取的pattern通过对称、旋转、转置等变换构造更多的pattern,构造出的pattern拥有与它们的来源相同的weight
#             temp_patterns, temp_weights = patterns[:], weights
#             weights = dict.fromkeys(range(len(patterns)), 0)
#             patterns = patterns
#             for pattern in temp_patterns:
#                 current_patterns = [pattern]  #current_patterns = [当前pattern，左右对称，上下左右对称，上下对称，转置，旋转90°，旋转270°，转置并旋转180°]
#                 if changable[0]:
#                     current_patterns.append(LR_Reflect(current_patterns[0]))
#                 if changable[1] and changable[0]:
#                     current_patterns.append(UD_Reflect(current_patterns[1]))
#                 if changable[1]:
#                     current_patterns.append(UD_Reflect(current_patterns[0]))
#                 if changable[2]:
#                     current_patterns.extend(Rotated(current_patterns[0]))
#                 for each in current_patterns:
#                     if each not in patterns:
#                         patterns.append(each)
#                         weights[patterns.index(each)] = temp_weights[temp_patterns.index(pattern)]
#                     else:
#                         weights[patterns.index(each)] += temp_weights[temp_patterns.index(pattern)]
#             return patterns, weights

#         # if sum(changable):
#         # 允许变换，对patterns进行拓展
#         # patterns, weights = extend(patterns, weights, changable)
#         # return matrix, patterns, weights
#         # self.patterns = patterns
#         # self.weights = weights
#         # self.matrix = matrix

#     def get_rules(self):
#         '''
#         如果不允许对称及旋转变换，即需要更严格的规则，则处理num_matrix得到规则

#         如果允许变换，则通过比对每个pattern在某个方向是否可以和其他pattern重合来得到规则，这样得到的规则可能含有样图中不包含的情况
#         '''
#         rules = [[set() for _ in range(self.count)] for _ in range(4)]
#         left = (-1, 0)
#         right = (1, 0)
#         up = (0, -1)
#         down = (0, 1)
#         directions = [left, right, up, down]
#         if not sum(self.changable):
#             # 不允许变换，基于样图得到规则
#             for x in range(self.width):
#                 for y in range(self.height):
#                     for d in range(4):
#                         x1 = x + directions[d][0]
#                         y1 = y + directions[d][1]
#                         if x1 >= 0 and x1 < self.width and y1 >= 0 and y1 < self.height:
#                             # rules[d][self.num_matrix[x][y]].add(self.num_matrix[x1][y1])
#                             rules[d][self.matrix[x][y]].add(self.matrix[x1][y1])
#         else:
#             # 允许变换，基于patterns得到规则
#             N = self.N

#             def connectable(pattern, other_pattern, d):
#                 # 检测两个pattern在给定方向能否连接
#                 for x in range(N):
#                     for y in range(N):
#                         x1 = x - directions[d][0]
#                         y1 = y - directions[d][1]
#                         if x1 >= 0 and x1 < N and y1 >= 0 and y1 < N and other_pattern[x1][y1] != pattern[x][y]:
#                             return False
#                 return True

#             for pattern in range(self.count):  #pattern均用它在patterns中的index表示
#                 for other_pattern in range(self.count):
#                     for d in range(4):
#                         if connectable(self.patterns[pattern], self.patterns[other_pattern], d):
#                             rules[d][pattern].add(other_pattern)
#         return rules
#         #[[{p1,p2,p3,...}(第0种模式),{...}(1),{...},...](left), [{...},{...},...](right), [...](up), [...](down)]


class ScanPattern1:
    """ 
    扫描输入中的模式及其出现的频率，收集不同模式之间的邻近关系。
    
    模式(pattern)是进行分析的最小单元，大小为N*N。扫描输入矩阵，记录其中的模式并编号，将输入矩阵转化为由模式对应的编号组成的矩阵，同时记录模式间的邻近关系。如果允许对模式进行对称操作，将可能得到输入矩阵中不存在的模式。新模式之间的邻近关系同样由对称操作扩展。如果要求分析所有可能的邻近关系，则需要在各模式间进行邻近匹配。
     
    主要属性：
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
                    self.rules.append([set(None) for _ in range(4)])
                    index += 1
                self.make_rule((x, y))

        self.changable = (LR_Reflect, UD_Reflect, Rotatable)  # (是否允许对模式做左右对称，上下对称，旋转) 准备删掉, 用symmetry来替代这部分的功能

        if not symmetry:
            self.symmetry_operate(symmetry)
        self.revpatterns = {index: pattern for pattern, index in self.patterns}
        self.count = len(self.patterns)

        if N > 1 and AllRules:
            self.make_allrule()  # 对所有pattern扫描匹配

    def get_pattern(self, entry, position):
        '''获取pattern'''
        x, y = position[0], position[1]
        N = self.N
        pattern = tuple(tuple(entry[x1][y:y + N]) for x1 in range(x, x + N))
        return pattern

    def make_rule(self, position):
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
        for pattern in prime_patterns:
            for sym in symmetry:
                p = symmetry_op[sym](pattern)
                try:
                    self.weights[prime_patterns[p]] += 1
                except KeyError:
                    self.patterns[p] = index
                    self.weights.append(1)
                    # self.rules.append([set(None) for _ in range(4)])
                    index += 1

    def make_allrule(self):
        assert self.N > 1

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

    # def build(self, entry):
    #     '''将输入的图像转为矩阵，并构建patterns和weights'''

    #     # patterns, revpatterns, weights = {}, {}, []
    #     # matrix = [[None] * self.height for _ in range(self.width)]
    #     N = self.N

    #     # index = 0
    #     # for x in range(self.width):
    #     #     for y in range(self.height):
    #     #         pattern = self.get_pattern(entry, (x, y))
    #     #         try:
    #     #             i = patterns[pattern]
    #     #             weights[i] += 1
    #     #             matrix[x][y] = i
    #     #         except KeyError:
    #     #             patterns[pattern] = index
    #     #             revpatterns[index] = pattern
    #     #             matrix[x][y] = index
    #     #             weights.append(1)
    #     #             index += 1

    #     def LR_Reflect(pattern):  #对pattern做左右对称
    #         return list(reversed(pattern))

    #     def UD_Reflect(pattern):  #对pattern做上下对称
    #         return [list(reversed(pattern[i])) for i in range(N)]

    #     def Rotated(pattern):
    #         #对pattern做旋转、转置，返回一个由转置、旋转90°、270°、转置并旋转180°后的pattern组成的list
    #         def transposition(pattern):  #将一个pattern转置
    #             temp = [[None] * N for _ in range(N)]
    #             for x in range(N):
    #                 for y in range(N):
    #                     temp[y][x] = pattern[x][y]
    #             return temp

    #         p1 = transposition(pattern)
    #         p2 = LR_Reflect(p1)
    #         p3 = UD_Reflect(p1)
    #         p4 = UD_Reflect(p2)
    #         return [p1, p2, p3, p4]

    # def extend(patterns, weights, changable):
    #     # 如果允许，基于图中提取的pattern通过对称、旋转、转置等变换构造更多的pattern,构造出的pattern拥有与它们的来源相同的weight
    #     temp_patterns, temp_weights = patterns[:], weights
    #     weights = dict.fromkeys(range(len(patterns)), 0)
    #     patterns = patterns
    #     for pattern in temp_patterns:
    #         current_patterns = [pattern]  #current_patterns = [当前pattern，左右对称，上下左右对称，上下对称，转置，旋转90°，旋转270°，转置并旋转180°]
    #         if changable[0]:
    #             current_patterns.append(LR_Reflect(current_patterns[0]))
    #         if changable[1] and changable[0]:
    #             current_patterns.append(UD_Reflect(current_patterns[1]))
    #         if changable[1]:
    #             current_patterns.append(UD_Reflect(current_patterns[0]))
    #         if changable[2]:
    #             current_patterns.extend(Rotated(current_patterns[0]))
    #         for each in current_patterns:
    #             if each not in patterns:
    #                 patterns.append(each)
    #                 weights[patterns.index(each)] = temp_weights[temp_patterns.index(pattern)]
    #             else:
    #                 weights[patterns.index(each)] += temp_weights[temp_patterns.index(pattern)]
    #     return patterns, weights

    # if sum(changable):
    # 允许变换，对patterns进行拓展
    # patterns, weights = extend(patterns, weights, changable)
    # return matrix, patterns, weights

    # self.patterns, self.revpatterns = patterns, revpatterns
    # self.weights = weights
    # self.matrix = matrix

    # def get_rules(self):
    #     '''
    #     如果不允许对称及旋转变换，即需要更严格的规则，则处理num_matrix得到规则

    #     如果允许变换，则通过比对每个pattern在某个方向是否可以和其他pattern重合来得到规则，这样得到的规则可能含有样图中不包含的情况
    #     '''
    #     rules = [[set() for _ in range(self.count)] for _ in range(4)]
    #     left = (-1, 0)
    #     right = (1, 0)
    #     up = (0, -1)
    #     down = (0, 1)
    #     directions = [left, right, up, down]
    #     if not sum(self.changable):
    #         # 不允许变换，基于样图得到规则
    #         for x in range(self.width):
    #             for y in range(self.height):
    #                 for d in range(4):
    #                     x1 = x + directions[d][0]
    #                     y1 = y + directions[d][1]
    #                     if x1 >= 0 and x1 < self.width and y1 >= 0 and y1 < self.height:
    #                         # rules[d][self.num_matrix[x][y]].add(self.num_matrix[x1][y1])
    #                         rules[d][self.matrix[x][y]].add(self.matrix[x1][y1])
    #     else:
    #         # 允许变换，基于patterns得到规则
    #         N = self.N

    #         def connectable(pattern, other_pattern, d):
    #             # 检测两个pattern在给定方向能否连接
    #             for x in range(N):
    #                 for y in range(N):
    #                     x1 = x - directions[d][0]
    #                     y1 = y - directions[d][1]
    #                     if x1 >= 0 and x1 < N and y1 >= 0 and y1 < N and other_pattern[x1][y1] != pattern[x][y]:
    #                         return False
    #             return True

    #         for pattern in range(self.count):  #pattern均用它在patterns中的index表示
    #             for other_pattern in range(self.count):
    #                 for d in range(4):
    #                     if connectable(self.patterns[pattern], self.patterns[other_pattern], d):
    #                         rules[d][pattern].add(other_pattern)
    #     return rules
    #     #[[{p1,p2,p3,...}(第0种模式),{...}(1),{...},...](left), [{...},{...},...](right), [...](up), [...](down)]


# i = ScanPattern(r'image_path', 3, True, True, True)
# patterns, weights, rules, N = i.patterns, i.weights, i.rules, i.N


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