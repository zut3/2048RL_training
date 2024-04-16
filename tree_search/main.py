import sys
from random import choice
from time import time
from math import log, sqrt
import logging

sys.path.append('..')

from game.Memory import Collector
from game.enums import Direction, TurnMove
from game.state import State

from tree import Node

logger = logging.getLogger(__name__)
logging.basicConfig(filename='collect.log', level=logging.DEBUG)


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
            state.random_spawn()

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


        while root.can_add_child():
            child = root.add_random_child()
            reward = self.simulate_game(child)
            
            child.record(reward)
            root.record(reward)

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

        best = None
        max_p = -1

        for ch in root.childs:
            if ch.get_stat() > max_p:
                best = ch.move
                max_p = ch.get_stat()


        return best, cnt
        
def collect_data(agent, collector, max_moves=1500):

    start = time()
    state = State().random_spawn()
    collector.begin_record()

    count = 0

    while count < max_moves and state.can_play():

        if state.can_play():
            move, _ = agent.select_move(state)

            collector.add(state.to_numpy(), int(move))

            state = state.apply_move(move)
            
        if state.can_play():
            state = state.random_spawn()        

        count += 1
        print(count)

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

        collect_data(agent, collector, 1)
        
        collector.serialize(f'./data/games_{collector_epoch}.h5')

        if len(collector) >= 10:
            collector_epoch += 1
            del collector 
            collector = Collector()
