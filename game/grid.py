import numpy as np
from enum import Enum


def stack(matrix):
    new = np.zeros_like(matrix)

    for i in range(matrix.shape[0]):
        pos = 0
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 0:
                continue
            
            new[i, pos] = matrix[i, j]
            pos += 1
    return new

def combine(matrix):
    score = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1] - 1):
            if(matrix[i, j] == matrix[i, j+1] and matrix[i, j] != 0):
                matrix[i, j] = 2 * matrix[i, j]
                matrix[i, j+1] = 0
                score += 2 * matrix[i, j].astype(np.int32)
    
    return matrix, score



class Grid(object):
    def __init__(self, w, h):
        self.weight = w
        self.height = h
        self._grid = np.zeros((w, h))
        self.score = 0

    @staticmethod
    def correct_move(prev, now):
        return (prev._grid != now._grid).any()

    def move_left(self):
        self._grid = stack(self._grid)
        
        self._grid, s = combine(self._grid)
        self.score += s

        self._grid = stack(self._grid)

    def move_right(self):
        self._grid = np.fliplr(self._grid)
        self.move_left()
        self._grid = np.fliplr(self._grid)
        
    def move_up(self):
        self._grid = self._grid.T
        self.move_left()
        self._grid = self._grid.T

    def move_down(self):
        self._grid = self._grid.T
        self.move_right()
        self._grid = self._grid.T
    
    def spawn(self, x, y):
        if self._grid[x, y] != 0:
            raise Exception('trying to spawn in not empty cell')
        
        self._grid[x, y] = 2
    
    def is_lose(self) -> bool:        
        copy = np.copy(self._grid)
        
        def _all_filled(x):
            s = stack(x)
            s, _ = combine(s)
            s = stack(s)

            return np.sum(s == 0) == 0
        
        return _all_filled(copy) and _all_filled(np.fliplr(copy)) \
                and _all_filled(copy.T) and _all_filled(np.fliplr(copy.T))

    def is_win(self) -> bool:
        return np.sum(self._grid == 2048) > 0

    def can_play(self) -> bool:
        return (not self.is_lose()) and (not self.is_win())
    
    def have_empty(self) -> bool:
        return np.sum(self._grid == 0) > 0
    
    def __repr__(self) -> str:
        return str(self._grid.astype(np.int32))



if __name__ == '__main__':
    print('===testing grid 5x5===')
    grid = Grid(5, 5)
    grid.spawn()
    print(grid, end='\n\n')
    grid.move_left()
    print(grid, end='\n\n')
    grid.move_right()
    print(grid, end='\n\n')


