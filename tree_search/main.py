import sys
from random import choice
from time import time
from math import log, sqrt

import numpy as np

sys.path.append('..')

from game.Memory import Collector
from game.enums import Direction, TurnMove
from game.state import State

from tree import Node


def uct_score(wins, total, current, temperature):
    exploration = sqrt(log(total) / current)
    return wins + temperature * exploration

class MonteCarloAgent:
    def __init__(self, duration=1) -> None:
        self.collector = Collector()   
        self.duration = duration

    def simulate_game(self, node: Node):
        state = node.state
        
        if state.move == TurnMove.Game:
            state.spawn()

        while state.can_play():
            move = choice(state.valid_moves())
            
            state = state.apply_move(Direction(move))
            if state.can_play():
                state = state.spawn()

        if state.is_win():
            print('Win!!')

        return state.is_win()
                
    def select_child(self, node: Node) -> Node:
        if len(node.childs) == 0:
            return node
        
        total_games = sum([c.n_games for c in node.childs])
        
        best = None
        max_score = -1
        for child in node.childs:
            score = uct_score(child.get_stat(), total_games, child.n_games, 1.05)
            if score > max_score:
                max_score = score
                best = child
    
        return best
            
    
    def select_move(self, state: State):
        if not state.can_play():
            raise Exception('game is over')

        root = Node(None, state)

        start = time()

        while (time() - start) < self.duration:
            node = root
            
            while (not node.can_add_child()) and (not node.is_leaf()):
                node = self.select_child(node)    

            if node.can_add_child():
                node = node.add_random_child()

            reward = self.simulate_game(node)

            while node is not None:
                node.record(reward)
                node = node.parent

        best = None
        max_p = -1

        for ch in root.childs:
            if ch.get_stat() > max_p:
                best = ch.move
                max_p = ch.get_stat()
        
        return best
        
def collect_data(agent, collector):

    start = time()
    state = State().spawn()
    collector.begin_record()

    count = 1
    while state.can_play():

        if state.can_play():
            move = agent.select_move(state)
            collector.add(state.to_numpy(), int(move))

            state = state.apply_move(move)
            
        if state.can_play():
            state = state.spawn()        

        print(count)
        count += 1

    end = time()

    collector.stop_record(1 if state.is_win() else -1)
    collector.serialize('data.h5')

    print("Win: ", state.is_win())
    print(state.grid)

    print((end - start) / 60, 'mins')


if __name__ == '__main__':
    agent = MonteCarloAgent(1.5)
    collector = Collector()

    for i in range(1):
        print(f'====epoch #{i+1}====\n\n')

        collect_data(agent, collector)
    