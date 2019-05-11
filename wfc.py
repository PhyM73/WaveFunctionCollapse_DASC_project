import math
import random
from PIL import Image

image = Image.open(r'image_path')
class Scan:

    def __init__(self, image):
        self.width = image.width
        self.height = image.height
        self.matrix = self.trans_image_to_matrix(image)
        self.tiles = set()  #{tile1, tile2, tile3, ...}
        self.weights = dict() #{tile1:weight, tile2:weight, ...}
        self.rules = dict()#self.rules -- {tile1:{left:{tile1:count, tile2:count,...}, right:{...}, up:{...}, down:{...}}, tile2:{...}, ...}

    def trans_image_to_matrix(self, image):
        #将输入的图片转化为矩阵，矩阵元为像素值tuple
        #暂时还只能将每个像素当作一个tile，可以尝试将图片分隔为由多个像素点组成的网格，每个网格当作一个tile
        matrix = [[None]*self.height for _ in range(self.width)]
        for x in range(self.width):
            for y in range(self.height):
                matrix[x][y] = image.getpixel((x, y))
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
        left=(-1,0); right=(1,0); up=(0,-1); down=(0,1)
        directions = {left, right, up, down}
        rule_in_one_dir = dict.fromkeys(self.tiles, 0)       
        
        for tile in self.tiles: 
            self.rules[tile] = {left:rule_in_one_dir.copy(), right:rule_in_one_dir.copy(), up:rule_in_one_dir.copy(), down:rule_in_one_dir.copy()}            
        for x in range(self.width):
            for y in range(self.height):
                for direction in directions:
                    x1 = x + direction[0]; y1 = y + direction[1]
                    if x1 >= 0 and x1 < self.width and y1 >= 0 and y1 < self.height:
                        self.rules[self.matrix[x][y]][direction][self.matrix[x1][y1]] += 1
        return self.tiles, self.weights, self.rules

#对于self.rules有两种想法，一种是一个字典，其keys对应每一种tile，其values为一个字典，记录了在它每个方向出现每种tile的次数，也就是上面实现的；
#另一种想法是返回一个字典，其keys对应一个list，为[current_tile, other_tile, direction]，其values为该模式出现的次数。
#对于其中记录的信息，上面的实现是记录了每种模式出现的次数，也可以改为记录它们出现的频率，视之后使用的方便调整。

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