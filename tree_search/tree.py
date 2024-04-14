from random import randint
from game.state import State
from game.enums import Direction

class Node:
    def __init__(self, parent, state: State, move=None):
        self.parent = parent
        self.state = state
        self.move = move

        self.unapplied_moves = state.valid_moves()
        self.childs = []

        self.rewards = 0
        self.n_games = 0

    def can_add_child(self):
        return len(self.unapplied_moves) > 0

    def add_random_child(self):
        if not self.can_add_child():
            raise Exception("can't add new child")

        idx = randint(0, len(self.unapplied_moves)-1)
        move = self.unapplied_moves.pop(idx)
        
        child = Node(self, self.state.apply_move(Direction(move)), Direction(move))
        self.childs.append(child)

        return child

    def record(self, reward):
        self.rewards += reward
        self.n_games += 1

    def get_stat(self):
        return self.rewards / self.n_games

    def is_leaf(self):
        return not self.state.can_play()