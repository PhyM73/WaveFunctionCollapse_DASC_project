import math


class Node():
    '''格点单元'''

    def __init__(self, state_space):
        self.space = state_space
        self.entropy = Node.shannon(state_space)

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
        self.width, self.height = size[0], size[1]
        self.wave = [[Node(state_space)] * size[0]] * size[1]

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
        pass
