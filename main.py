import numpy as np

from random import choice
from time import time
from math import log, sqrt

import sys
import traceback
import logging

from game.Memory import Collector
from game.enums import Direction, TurnMove
from game.state import State

from tree import Node

logging.basicConfig(filename='collect.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                              "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)

def exp_handler(t, value, tb):
    logger.exception("Get exception: {0}".format(str(value)))
    
    all_tb = "\n"
    for e in traceback.format_tb(tb):
        all_tb += e

    logger.info(all_tb)

sys.excepthook = exp_handler

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
            state = state.random_spawn()

        while state.can_play():
            move = choice(state.get_valid_moves())
            
            state = state.apply_move(Direction(move))
            if state.can_play():
                state = state.random_spawn()

        if state.is_win():
            logger.debug('Win!!')
            logger.debug(state.grid)

        return state.is_win()
                
    def select_child(self, node: Node) -> Node:
        if len(node.childs) == 0:
            return node
        
        total_games = sum([c.n_games for c in node.childs])
        
        best = None
        max_score = -1
        for child in node.childs:
            score = uct_score(child.get_stat(), total_games, child.n_games, 1.5)
            if score > max_score:
                max_score = score
                best = child

        return best
            
    
    def select_move(self, state: State):
        if not state.can_play():
            raise Exception('game is over')

        root = Node(None, state)

        st = time()
    
        while root.can_add_child():
            ch = root.add_random_child()
            reward = self.simulate_game(ch)
            
            ch.record(reward)
            root.record(reward)

        print('All moves computation(mins) ', (time() - st) / 60)

        cnt = 0

        start = time()

        while (time() - start) < self.duration:
            node = root
        
            while (not node.can_add_child()) and (not node.is_leaf()):
                node = self.select_child(node) 
                if node != root:
                    cnt += 1

            if node.can_add_child():
                node = node.add_random_child()

            reward = self.simulate_game(node)

            while node is not None:
                node.record(reward)
                node = node.parent

        logger.debug(f"utc's {cnt}")

        mean_reward = {}

        for ch in root.childs:
            if not mean_reward.get(ch.move):
                mean_reward[ch.move] = [ch.get_stat(), ]
            else:
                mean_reward[ch.move].append(ch.get_stat())

        maxi = -1
        best = None

        for k, v in mean_reward.items():
            m = sum(v) / len(v)
            
            if m > maxi:
                maxi = m
                best = k
        
        return best
        
def collect_data(agent, collector, max_moves=1e9):

    start = time()
    state = State().random_spawn()
    collector.begin_record()

    count = 0

    while state.can_play() and count < max_moves:
            
        if state.can_play():
            move = agent.select_move(state)

            collector.add(state.to_numpy(), int(move))

            state = state.apply_move(move)
            
        if state.can_play():
            state = state.random_spawn()        

        count += 1
        logger.info(f'move #{count}')

    end = time()

    collector.stop_record(1 if state.is_win() else -1)
    
    logger.debug(f"Win: {state.is_win()}")
    logger.debug(state.grid)

    logger.debug(f"{(end - start) / 60} mins")
    

if __name__ == '__main__':
    agent = MonteCarloAgent(2)  
    collector = Collector()

    collector_epoch = 0

    for i in range(1):  
        logger.info(f'starting epoch #{i}')

        collect_data(agent, collector)
        
        collector.serialize(f'./data/games_{i}.h5')

        del collector 
        collector = Collector()
