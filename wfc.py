import math
import random


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
    
    Attributes：
        wave: A 2-D matrix which contains all the Knots.
        size: The shape of the wave.

        options:  All the optional functions.
        patterns: The index => pattern pairs.     
        weights:  Record the weight corresponding to the index of each pattern 
        rules:    Record patterns that each patterns can match in each directions.
     
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
        # Scan the entry and extract patterns and rules.
        self.N = N
        self.options = {
            'PeriIpt': PeriodicInput,
            'PeriOpt': PeriodicOutput,
            'Rot': Rotation,
            'Ref': Reflection,
            'surv': surveil
        }
        self.patterns, self.weights, self.rules = {}, [], []
        self.BuildPatterns(entry)
        self.patterns = {index: pattern for pattern, index in self.patterns.items()}
        if N > 1 and AllRules:
            self.make_all_rules()


        self.size = (size[0] - N + 1, size[1] - N + 1)
        self.Stack = []

        # Initializes the 2-D WaveFunction matrix, each Knot of the matrix
        # starts with all states as possible. No state is forbidden yet.
        # And every Knot is waited to collapse.
        state_space = {state: self.weights[state] for state in self.patterns.keys()}
        self.wave = [[Knot(state_space.copy()) for i in range(self.size[1])] for j in range(self.size[0])]
        self.wait_to_collapse = set((x, y) for x in range(self.size[0]) for y in range(self.size[1]))

    @staticmethod
    def symmetry(m, reflect, rotate):
        """Returns matrix which performed symmetry operations."""

        def LR_reflect(m):
            return tuple(reversed(m))

        def UD_reflect(m):
            return tuple(tuple(reversed(m[x][:])) for x in range(len(m)))

        operand = {m}
        if reflect:
            operand = operand | {LR_reflect(m), UD_reflect(m), LR_reflect(UD_reflect(m))}
        if rotate:
            m1 = tuple(tuple(m[y][x] for y in range(len(m[0]))) for x in range(len(m)))
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
        """Makes the adjacent-rule for the pattern at the location with 
        patterns at its left side and its top side."""
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
                            return self.backtrack()

                        elif self.Stack and (nb not in self.Stack[-1].keys()):
                            # Push this changed Knot into the Stack.
                            self.Stack[-1][nb] = self[nb].space.copy()
                        self[nb] = Knot({state: self.weights[state] for state in available})
                        PropagStack.append(nb)
                        changed.add(nb)
        return changed

    def backtrack(self):
        """Backtracks to the previous step. 
        If there is no way to backtrack then this method raises CollapseError. """
        print('0')
        if len(self.Stack):
            step = self.Stack.pop()
            # Restore all the Knot affected by the last collapse.
            for (position, space) in step.items():
                self[position] = Knot(space)
                self.wait_to_collapse.add(position)
            return set(step.keys())
        else:
            raise CollapseError("No Sulotion")

    def observe(self, surveil):
        '''Observe the whole WaveFunction'''
        try:
            if surveil:
                while self.wait_to_collapse:
                    yield self.collapse(self.min_entropy_pos())
            else:
                while self.wait_to_collapse:
                    list(self.collapse(self.min_entropy_pos()))
                yield [(x, y) for x in range(self.size[0]) for y in range(self.size[1])]
        except CollapseError:
            raise CollapseError

if __name__ == "__main__":
    def MatrixProcessor(entry, size, N, options):
        def update(matrix, position, w, N):
            limit_i = N if position[0] == w.size[0] - 1 else 1
            limit_j = N if position[1] == w.size[1] - 1 else 1
            for i in range(limit_i):
                for j in range(limit_j):
                    matrix[position[0] + i][position[1] + j] = w.patterns[list(w[position].space.keys())[0]][i][j]
            return matrix

        w = WaveFunction(size, entry, N=N, **options)
        matrix = [[None] * size[0] for _ in range(size[1])]

        for changed in w.observe(w.options['surv']):
            for pos in changed:
                matrix = update(matrix, pos, w, N)
        for eachline in matrix:
            print(eachline)

    # This is a demo to show a how wfc is working on string matrix.
    # Imagine that below is a map, where 'L' = land, 'C' = coast, 'S' = sea,
    # and now we will show you that how wfc generate maps with some natural rules, such as land cannot be next to sea.
    entry = (
        ('L', 'L', 'L', 'L', 'L', 'L', 'L'),
        ('L', 'L', 'C', 'L', 'L', 'L', 'L'),
        ('C', 'C', 'S', 'C', 'C', 'L', 'L'),
        ('S', 'S', 'S', 'S', 'S', 'C', 'L'),
        ('S', 'S', 'S', 'S', 'S', 'S', 'C')
    )
    options = {        
        'AllRules':0,
        'surveil':0,
        'PeriodicInput':0,
        'PeriodicOutput':0,
        'Rotation':0,
        'Reflection':0
    }
    MatrixProcessor(entry, (10,10),N=1,options=options)