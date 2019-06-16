import copy
import time

class Cell:
    def __init__(self, line_x, line_y, label):        
        self.wave = {1,2,3,4,5,6,7,8,9}
        for each in line_x:
            if each in self.wave:
                self.wave.remove(each)
        for each in line_y:
            if each in self.wave:
                self.wave.remove(each)
        for each in label:
            if each in self.wave:
                self.wave.remove(each)
        self.count = len(self.wave)
    def pop(self):
        self.count -= 1
        return self.wave.pop()
    def remove(self, item):
        if item in self.wave:
            self.count -= 1
            self.wave.remove(item)

class Sudoku:
    def __init__(self, array):
        self.i = 0 #循环次数
        self.num = 0 #解的个数
        self.solvable = False
        self.result = array
        self.process = [] #作为一个栈
        self.array = [[] for _ in range(9)] #记录每点可选的值
        for x in range(9):
            for y in range(9):
                if self.result[x][y] != 0:
                    self.array[x].append(0)
                else: 
                    label = []
                    for x1 in range(x-x%3, x-x%3+3):
                        for y1 in range(y-y%3, y-y%3+3):
                            label.append(self.result[x1][y1])
                    self.array[x].append(Cell(self.result[x], [self.result[i][y] for i in range(9)], label))
    
    def find_min(self):
        choice = -1
        pos = [-1,-1]
        for x in range(9):
            for y in range(9):
                if self.array[x][y] == 0:
                    continue
                elif choice == -1 or choice > self.array[x][y].count:
                    choice = self.array[x][y].count
                    pos = [x, y]
        return [choice, pos] 

    def influnence(self, num, x ,y):
        self.array[x][y] = 0
        label = []
        for x1 in range(x-x%3, x-x%3+3):
            for y1 in range(y-y%3, y-y%3+3):
                label.append([x1, y1])
        for x1 in range(9):
           if x1 != x and self.array[x1][y] != 0:
                self.array[x1][y].remove(num)
        for y1 in range(9):
            if y1 != y and self.array[x][y1] != 0:
                self.array[x][y1].remove(num)
        for pos in label:
            x1 = pos[0]
            y1 = pos[1]
            if (x1 != x or y1 != y) and self.array[x1][y1] != 0:
                self.array[x1][y1].remove(num)

    def go_back(self):
        back = self.process.pop()
        self.result = back[0]
        self.array = back[1]
        
    def fill(self):
        gap = self.find_min()
        while gap[0] != -1:
            self.i += 1
            x = gap[1][0]
            y = gap[1][1]
            if gap[0] == 0:
                if self.process != []:
                    self.go_back()
                elif self.solvable == False:
                    print("No Solution")
                    return self.solvable
            elif gap[0] > 1:
                step = [copy.deepcopy(self.result)]
                self.result[x][y] = self.array[x][y].pop()
                step.append(copy.deepcopy(self.array))
                self.process.append(step)
                self.influnence(self.result[x][y], x, y)
            elif gap[0] == 1:
                self.result[x][y] = self.array[x][y].pop()
                self.influnence(self.result[x][y], x, y)
            gap = self.find_min()
        for eachline in self.result:
            print(eachline)
        print("%d Steps" % self.i)
        self.solvable = True
        return True

    def fill_all(self):
        gap = self.find_min()
        while gap[0] != -1 or self.process != []:
            self.i += 1
            x = gap[1][0]
            y = gap[1][1]
            if gap[0] == -1:
                for eachline in self.result:
                    print(eachline)
                print("%d Steps" % self.i)
                self.solvable = True
                self.num += 1
                self.go_back()        
            elif gap[0] == 0:
                if self.process != []:
                    self.go_back()
                elif self.solvable == False:
                    print("No Solution")
                    return self.solvable                
                elif self.solvable:
                    print("%d Solutions" % self.num)
                    print("Totally %d Steps" % self.i)
                    return self.solvable                
            elif gap[0] > 1:
                step = [copy.deepcopy(self.result)]
                self.result[x][y] = self.array[x][y].pop()
                step.append(copy.deepcopy(self.array))
                self.process.append(step)
                self.influnence(self.result[x][y], x, y)
            elif gap[0] == 1:
                self.result[x][y] = self.array[x][y].pop()
                self.influnence(self.result[x][y], x, y)
            gap = self.find_min()
        print("%d Solutions" % self.num)
        print("Totally %d Steps" % self.i)
        return True
''' w = Sudoku([
    [0,0,1,4,0,7,0,0,0],
    [0,0,0,0,0,0,9,8,0],
    [3,0,8,0,0,0,0,0,6],
    [1,0,0,3,0,0,5,0,0],
    [0,0,0,1,0,2,0,0,0],
    [0,0,6,0,0,5,0,0,7],
    [7,0,0,0,0,0,2,0,9],
    [0,9,3,0,0,0,0,0,0],
    [0,0,0,8,0,9,6,0,0]
]) '''
w = Sudoku([
    [0,0,1,0,0,7,0,0,0],
    [0,0,0,0,0,0,9,8,0],
    [3,0,8,0,0,0,0,0,6],
    [1,0,0,3,0,0,5,0,0],
    [0,0,0,1,0,2,0,0,0],
    [0,0,6,0,0,5,0,0,7],
    [7,0,0,0,0,0,2,0,9],
    [0,9,3,0,0,0,0,0,0],
    [0,0,0,8,0,9,6,0,0]
])
""" w = Sudoku([
    [0,0,5,3,0,0,0,0,0],
    [8,0,0,0,0,0,0,2,0],
    [0,7,0,0,1,0,5,0,0],
    [4,0,0,0,0,5,3,0,0],
    [0,1,0,0,7,0,0,0,6],
    [0,0,3,2,0,0,0,8,0],
    [0,6,0,5,0,0,0,0,9],
    [0,0,4,0,0,0,0,3,0],
    [0,0,0,0,0,9,7,0,0]
])#the most difficult sudoku """
#t1 = time.process_time()
t1 = time.perf_counter()
#w.fill()
w.fill_all()
#t2 = time.process_time()
t2 = time.perf_counter()
print("Spend %f s" % (t2 - t1))