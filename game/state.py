import numpy as np
from copy import deepcopy
from game.grid import Grid
from game.enums import Direction, TurnMove

from random import choice

WIDTH = 5

class State:
    def __init__(self, grid=None, move=TurnMove.Game) -> None:
        self.grid = grid
        if grid is None:
            self.grid = Grid(5, 5)

        self.move = move
        self._play = self.grid.can_play()
        self._valid_game_moves = np.argwhere(self.grid._grid == 0)

    def random_spawn(self):
        if not self.can_play():
            raise Exception('game is over')

        x, y = choice(self.get_valid_game_moves())
        return self.spawn(x, y)

    def spawn(self, x, y):
        if not self.can_play():
            raise Exception('game is over')

        new_grid = deepcopy(self.grid)
        new_grid.spawn(x, y)

        return State(new_grid, move=TurnMove.Player)
        
    def apply_move(self, dir: Direction):
        if not self.can_play():
            raise Exception('game is over')

        new_grid = deepcopy(self.grid)
        assert new_grid is not self.grid

        if dir == Direction.up:
            new_grid.move_up()
        elif dir == Direction.down:
            new_grid.move_down()
        elif dir == Direction.left:
            new_grid.move_left()
        elif dir == Direction.right:
            new_grid.move_right()
        else:
            raise Exception('incorrect move')

        return State(new_grid, move=TurnMove.Game)

    def to_numpy(self):
        return self.grid._grid
    
    def get_valid_moves(self):
        moves = []

        for i in range(4):
            if Grid.correct_move(self.grid, self.apply_move(Direction(i)).grid):
                moves.append(i)

        return moves
    
    def get_valid_game_moves(self):
        return self._valid_game_moves
    
    def can_play(self):
        return self._play
    
    def is_win(self):
        return self.grid.is_win()
                         

# deprecated
class Game(object):
    def __init__(self, state=None):
        
        if state is None:
            self.grid = Grid(WIDTH, WIDTH)
            self.grid.spawn()
        else:
            self.grid = state


    def get_state(self):
        return self.grid._grid.copy()
    
    def get_score(self):
        return self.grid.score
    
    def is_win(self):
        return self.grid.is_win()

    def can_play(self):
        return self.grid.can_play()
    
    def corect_last_move(self):
        return self.grid.corect
    
    def spawn(self, x=None, y=None):
        if (x is None) and (y is None):
            self.grid.spawn()
        else:
            assert self.grid._grid[x, y] != 0
            self.grid._grid[x, y] = 2

    def act(self, dir: Direction, spawn=True):
        if not self.can_play():
            return

        if dir == Direction.up:
            self.grid.move_up()
        elif dir == Direction.down:
            self.grid.move_down()
        elif dir == Direction.left:
            self.grid.move_left()
        elif dir == Direction.right:
            self.grid.move_right()

        if self.grid.have_empty() and spawn:
            self.spawn()
