import math
import random


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
        return math.log(ws) - sum(
            map(lambda x: x * math.log(x), statespace.value())) / ws


class Wave():
    '''体系波函数'''

    def __init__(self, size, state_space):
        # 波函数为包含所有格点的矩阵
        self.width, self.height = size[0], size[1]
        self.wave = [[Lattice(state_space)] * size[0]] * size[1]

    def __getitem__(self, index):
        return self.wave[index[0]][index[1]]

    def min_entropy_pos(self):
        '''寻找熵最小的格点的位置'''
        x, y = 0, 0
        min_entropy = self.wave[x][y].entropy
        for i in range(self.width):
            for j in range(self.height):
                if self.wave[x][y].entropy == 0:
                    continue
                if self.wave[i][j].entropy < min_entropy:
                    x, y = i, j
                    min_entropy = self.wave[x][y].entropy
        return x, y

    def collapse(self):
        '''选择目前熵最小的格点，并随机塌缩'''
        x, y = self.min_entropy_pos()
        if len(self.wave[x][y].space) == 1:
            self.wave[x][y].entropy == 0
        elif len(self[x, y].space) < 1:
            s = random.choices(self[x, y].space.keys(),
                               weights=self[x, y].space.value())
            self.propagate((x, y))
        else:
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