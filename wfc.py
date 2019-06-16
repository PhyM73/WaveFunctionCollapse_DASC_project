import math
import random
import numpy as np

import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib
import tkinter.filedialog


class Knot():
    """Knot object which contains its state space and Shannon entropy."""

    def __init__(self, state_space):
        self.space = state_space  # state_space = {state:weight}
        self.entropy = Knot.shannon(state_space)

    @staticmethod
    def shannon(state_space):
        """Returns the Shannon entropy of the `state_space`."""
        if isinstance(state_space, int) or len(state_space) == 1:
            return 0
        ws = sum(state_space.values())
        if ws == 0:
            print(state_space)
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
        wave: A 2-D matrix which contains all the Knots.

        patterns: {i1:p1, i2:p2, ...}     
        weights: [w1, w2, ...]     
        rules: [ [{left}, {right}, {up}, {down}], [... ], ... ]        记录每个模式在各个方向上可以匹配的模式
     
    """

    def __init__(self,
                 size,
                 entry,
                 N=3,
                 *,
                 surveil=True,
                 AllRules=False,
                 Rotation=False,
                 Reflection=False,
                 PeriodicInput=False,
                 PeriodicOutput=False):
        # 初始化patterns
        self.N = N
        self.options = {'PeriIpt': PeriodicInput, 'PeriOpt': PeriodicOutput, 'Rot': Rotation, 'Ref': Reflection}
        self.patterns, self.weights, self.rules = {}, [], []
        self.BuildPatterns(entry)
        self.patterns = {index: pattern for pattern, index in self.patterns.items()}
        if N > 1 and AllRules:
            self.make_all_rules()

        self.size = (size[0] - N + 1, size[1] - N + 1)
        self.wait_to_collapse = set((x, y) for x in range(self.size[0]) for y in range(self.size[1]))
        self.Stack = []

        # Initializes the 2-D WaveFunction matrix, each Knot of the matrix
        # starts with all states as possible. No state is forbidden yet.
        state_space = {state: self.weights[state] for state in self.patterns.keys()}
        self.wave = [[Knot(state_space.copy()) for i in range(self.size[1])] for j in range(self.size[0])]

    @staticmethod
    def symmetry(m, reflect, rotate):

        def LR_reflect(m):
            return tuple(reversed(m))

        def UD_reflect(m):
            return tuple(reversed(m[x][:]) for x in range(len(m)))

        operand = {m}
        if reflect:
            operand = operand | {LR_reflect(m), UD_reflect(m), LR_reflect(UD_reflect(m))}
        if rotate:
            m1 = tuple(tuple(m[y][x] for y in range(m[0])) for x in range(m))
            operand = operand | {LR_reflect(m1), UD_reflect(m1), LR_reflect(UD_reflect(m1))}

        return operand

    def BuildPatterns(self, entry):
        """Parses the `entry` matrix. Extracts patterns, weights and adjacent rules. """
        N = self.N
        for ent in WaveFunction.symmetry(entry, self.options['Ref'], self.options['Rot']):
            index = len(self.patterns)

            if self.options['PeriIpt']:
                width, height = len(ent) - 1, len(ent[0]) - 1
                ent = [ent[x][:] + ent[x][:N - 1] for x in range(len(ent))]
                ent = ent[:] + ent[:N - 1]
            else:
                width, height = len(ent) - N + 1, len(ent[0]) - N + 1

            matrix = [[None] * height for _ in range(width)]
            for x in range(width):
                for y in range(height):
                    # Extract an N*N matrix as a pattern with the upper left corner being (x, y).
                    pat = tuple(tuple(ent[x1][y:y + N]) for x1 in range(x, x + N))

                    # If this pattern already exists, simply increment its weight. Otherwise, records
                    # the new pattern and initializes its weight as 1, then increment the pattern index.
                    try:
                        matrix[x][y] = self.patterns[pat]
                        self.weights[matrix[x][y]] += 1
                    except KeyError:
                        self.patterns[pat] = matrix[x][y] = index
                        self.weights.append(1)
                        self.rules.append([set() for _ in range(4)])
                        index += 1
                    self.make_rule((x, y), matrix)

    def make_rule(self, position, matrix):
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

    def __getitem__(self, index):
        return self.wave[index[0]][index[1]]

    def __setitem__(self, index, value):
        self.wave[index[0]][index[1]] = value

    def min_entropy_pos(self):
        """Returns the position of the Knot whose statespace has the lowest entropy."""
        min_entropy = float("inf")
        for Knot in self.wait_to_collapse:
            noise = random.random() / 1000
            # Add some noise to mix things up a little
            if self[Knot].entropy - noise < min_entropy:
                position = Knot[:]
                min_entropy = self[position].entropy - noise
        return position

    def neighbor(self, position):
        """Yields neighboring Knots and their directions of a given `position`."""
        if self.options['PeriOpt']:
            if position[0] == 0:
                yield (self.size[0] - 1, position[1]), 0
            elif position[0] == self.size[0] - 1:
                yield (0, position[1]), 1
            if position[1] == 0:
                yield (position[0], self.size[1] - 1), 2
            elif position[1] == self.size[1] - 1:
                yield (position[0], 0), 3

        if position[0] > 0:
            yield (position[0] - 1, position[1]), 0
        if position[0] < self.size[0] - 1:
            yield (position[0] + 1, position[1]), 1
        if position[1] > 0:
            yield (position[0], position[1] - 1), 2
        if position[1] < self.size[1] - 1:
            yield (position[0], position[1] + 1), 3

    def collapse(self, position):
        """Collapses the Knot at `position`, and then propagates the consequences. """
        (x, y) = position
        if len(self[position]) < 1:
            # yield from self.backtrack()
            return self.backtrack()
        else:
            if len(self[position]) > 1:
                # Choose one possible pattern randomly and push this changed Knot into the Stack.
                states, w = list(self[position].space.keys()), list(self[position].space.values())
                elem = random.choices(states, weights=w)[0]
                del self.wave[x][y].space[elem]
                self.Stack.append({position: self[position].space.copy()})
                self[x, y] = Knot({elem: 1})
            self.wait_to_collapse.remove(position)
            return self.propagate(position)
            # yield position
            # yield from self.propagate(position)
            # self.propagate(position)

    def propagate(self, position):
        """Propagates the consequences of the wavefunction collapse or statespace 
        changing at `position`.This method keeps propagating the consequences of 
        the consequences,and so on until no consequences remain. 
        """
        PropagStack = [position]
        changed = {position}

        while PropagStack:
            pos = PropagStack.pop()

            for nb, direction in self.neighbor(pos):
                if nb in self.wait_to_collapse:
                    available = set.union(*[self.rules[state][direction] for state in self[pos].space.keys()])
                    if not set(self[nb].space.keys()).issubset(available):
                        available = available & set(self[nb].space.keys())
                        if len(available) == 0:
                            # print('no')
                            return self.backtrack()
                            # break

                        elif self.Stack and (nb not in self.Stack[-1].keys()):
                            # push this changed Knot into the Stack.
                            self.Stack[-1][nb] = self[nb].space.copy()
                        self[nb] = Knot({state: self.weights[state] for state in available})
                        PropagStack.append(nb)
                        # yield nb
                        # self.update(nb)
                        changed.add(nb)
        return changed

    def backtrack(self):
        """Backtracks to the previous step. 
        If there is no way to backtrack then this method raises CollapseError. """
        if len(self.Stack):
            step = self.Stack.pop()
            # Restore all the Girds affected by the last collapse
            for (position, space) in step.items():
                self[position] = Knot(space)
                self.wait_to_collapse.add(position)
                # yield position
            # yield from [step.keys()]
            # print('back')
            return set(step.keys())
        else:
            raise CollapseError("No Sulotion")

    def observe(self, surveil):
        '''Observe the whole WaveFunction'''
        if surveil:
            while self.wait_to_collapse:
                yield self.collapse(self.min_entropy_pos())
        else:
            while self.wait_to_collapse:
                list(self.collapse(self.min_entropy_pos()))
            yield [(x, y) for x in range(self.size[0]) for y in range(self.size[1])]


def image2matrix(image_path):
    """Convert image at `image_path` to matrix."""
    im = matplotlib.image.imread(image_path)
    img = tuple(tuple(tuple(im[x][y]) for y in range(im.shape[1])) for x in range(im.shape[0]))
    return img


def mean_pixel(wave, position, i, j):
    """Get the weighted mean of the state space of position as the pixel there"""
    keys, values = list(wave[position].space.keys()), np.array(list(wave[position].space.values()))
    return tuple(
        map(lambda x: np.average(np.array(x), weights=values), zip(*(wave.patterns[index][i][j] for index in keys))))


def ImageProcessor(image_path, size, N, options):
    #    AllRules=False,
    #    PeriodicInput=False,
    #    surveil=True,
    #    PeriodicOutput=False,
    #    Save=True,
    #    Reflection=False,
    #    Rotation=False,):
    entry = image2matrix(image_path)

    def update(matrix, position, w, N):
        limit_i = N if position[0] == w.size[0] - 1 else 1
        limit_j = N if position[1] == w.size[1] - 1 else 1
        for i in range(limit_i):
            for j in range(limit_j):
                matrix[position[0] + i, position[1] + j] = mean_pixel(w, position, i, j)
        return matrix

    # w = WaveFunction(size, entry, N=N, AllRules=AllRules, Rotation=Rotation, Reflection=Reflection)
    w = WaveFunction(size, entry, N=N, **options)
    fig = plt.figure(figsize=(8, 8))
    matrix = np.array([[mean_pixel(w, (0, 0), 0, 0)] * size[1] for _ in range(size[0])])
    im = plt.imshow(matrix)
    plt.axis('off')
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8, hspace=0.2, wspace=0.3)
    plt.pause(0.0001)

    for changed in w.observe(surveil):
        for pos in changed:
            matrix = update(matrix, pos, w, N)
        im.set_array(matrix)
        fig.canvas.draw()
        plt.pause(0.0001)
    # if Save: fig.savefig('result\\final.png', dpi=300, format='png')
    # top = Toplevel()
    # top.title("message")
    # msg = Message( top, text = "Done")
    # msg.pack()
    plt.show()


#################################################3
# ImageProcessor(r"samples\Cats.png", (50, 50), N=4, surveil=False, Periodic=True)
root = tk.Tk()
root.title("WaveFunctionCollapse")
root.geometry("600x400")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

tk.Label(frame, text='N:').grid(row=0, column=0, sticky=tk.W)
set_N = tk.Scale(frame, from_=1, to=4, length=100, tickinterval=1, orient=tk.HORIZONTAL)
set_N.grid(row=0, column=1, sticky=tk.W, columnspan=3)

tk.Label(frame, text='width:').grid(row=1, column=0, sticky=tk.W)
set_width = tk.Scale(frame, from_=10, to=100, length=300, tickinterval=10, orient=tk.HORIZONTAL)
set_width.grid(row=1, column=1, columnspan=3)

tk.Label(frame, text='height:').grid(row=2, column=0, sticky=tk.W)
set_height = tk.Scale(frame, from_=10, to=100, length=300, tickinterval=10, orient=tk.HORIZONTAL)
set_height.grid(row=2, column=1, columnspan=3)

tk.Label(frame, text='parameters:').grid(row=3, column=0, sticky=tk.W)

AL = tk.LabelFrame(frame, text="AllRules")
AL.grid(row=3, column=1, pady=30)
AllRules = tk.BooleanVar()
AllRules.set(False)
tk.Radiobutton(AL, text='False', variable=AllRules, value=False).pack()
tk.Radiobutton(AL, text='True', variable=AllRules, value=True).pack()

PL = tk.LabelFrame(frame, text="PeriodicInput")
PL.grid(row=3, column=2, pady=30)
PeriodicInput = tk.BooleanVar()
PeriodicInput.set(False)
tk.Radiobutton(PL, text='False', variable=PeriodicInput, value=False).pack()
tk.Radiobutton(PL, text='True', variable=PeriodicInput, value=True).pack()

PL = tk.LabelFrame(frame, text="PeriodicOutput")
PL.grid(row=3, column=5, pady=30)
PeriodicOutput = tk.BooleanVar()
PeriodicOutput.set(False)
tk.Radiobutton(PL, text='False', variable=PeriodicOutput, value=False).pack()
tk.Radiobutton(PL, text='True', variable=PeriodicOutput, value=True).pack()

SL = tk.LabelFrame(frame, text="Surveil")
SL.grid(row=3, column=3, pady=30)
surveil = tk.BooleanVar()
surveil.set(True)
tk.Radiobutton(SL, text='False', variable=surveil, value=False).pack()
tk.Radiobutton(SL, text='True', variable=surveil, value=True).pack()

SaveL = tk.LabelFrame(frame, text="Save")
SaveL.grid(row=3, column=4, pady=30)
save = tk.BooleanVar()
save.set(True)
tk.Radiobutton(SaveL, text='False', variable=save, value=False).pack()
tk.Radiobutton(SaveL, text='True', variable=save, value=True).pack()

path = tk.StringVar()
path.set('')


def get_image():
    path.set(tkinter.filedialog.askopenfilename())
    return True


options = {
    'AllRules': AllRules.get(),
    'surveil': surveil.get(),
    'PeriodicInput': PeriodicInput.get(),
    'PeriodicOutput': PeriodicOutput.get()
}


def main():
    ImageProcessor(path.get(), (set_height.get(), set_width.get()), N=set_N.get(), options=options)
    # ImageProcessor(path.get(), (set_height.get(), set_width.get()),
    #                N=set_N.get(),
    #                AllRules=AllRules.get(),
    #                surveil=surveil.get(),
    #                PeriodicInput=PeriodicInput.get(),
    #                Save=save.get(),
    #                PeriodicOutput=PeriodicOutput.get())


tk.Button(frame, text="open file", command=get_image).grid(row=4, column=1)
tk.Button(frame, text="begin", command=main).grid(row=4, column=2)

tk.mainloop()