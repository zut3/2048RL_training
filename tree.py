from random import randint
from game.state import State
from game.enums import Direction


class Node:
    def __init__(self, parent, state: State, move=None):
        self.parent = parent
        self.state = state
        self.move = move

        self.not_visited_states = []
        
        self.childs = []

        self.rewards = 0
        self.n_games = 0

        if not self.state.can_play():
            return    

        for i in self.state.get_valid_moves():
            s = self.state.apply_move(Direction(i))
            
            if not s.can_play():
                self.not_visited_states.append((s, i))
                continue

            for x, y in s.get_valid_game_moves():
                self.not_visited_states.append((s.spawn(x, y), i))

    def can_add_child(self):
        return len(self.not_visited_states) > 0

    def add_all_childs(self):
        length = len(self.not_visited_states)
        
        for i in range(length):
            new_state, move = self.not_visited_states[i]
            
            child = Node(self, new_state, Direction(move))
            self.childs.append(child)
        
        self.not_visited_states = []


    def add_random_child(self):
        if not self.can_add_child():
            raise Exception("can't add new child")

        idx = randint(0, len(self.not_visited_states)-1)
        new_state, move = self.not_visited_states.pop(idx)
        
        child = Node(self, new_state, Direction(move))
        self.childs.append(child)

        return child

    def record(self, reward):
        self.rewards += reward
        self.n_games += 1

    def get_stat(self):
        return self.rewards / self.n_games

    def is_leaf(self):
        return not self.state.can_play()