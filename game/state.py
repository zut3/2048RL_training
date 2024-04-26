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
            self.grid = Grid(WIDTH, WIDTH)

        self.move = move
        self._play = self.grid.can_play()

        self._valid_game_moves = []

        if self.can_play():
            self._valid_game_moves = np.argwhere(self.grid._grid == 1)
            

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
        if not self.can_play():
            return []
        
        moves = []

        for i in range(4):
            s = self.apply_move(Direction(i))
            
            if Grid.correct_move(self.grid, s.grid):
                moves.append(i)

        return moves
    
    def get_valid_game_moves(self):
        return self._valid_game_moves
    
    def can_play(self):
        return self._play
    
    def is_win(self):
        return self.grid.is_win()
                         
def get_reward(state: State):
    if not state.can_play() and state.is_win():
        return 6e3

    maxi = np.max(state.to_numpy())
    reward = maxi * np.sum(state.to_numpy() == maxi)
    reward /= np.var(state.to_numpy() != 1)

    return reward