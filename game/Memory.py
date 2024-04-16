import numpy as np
import h5py

class Buffer:
    def __init__(self):
        self.mem = []
    
    def __len__(self):
        return len(self.mem)

    def add(self, states, actions, rewards):
        self.mem += zip(states, actions, rewards) 

    def empty(self):
        self.mem.clear()
    
    def topk(self, k):
        if k > len(self.mem):
            raise ValueError("k greated than len of mem")
        
        def _s(x):
            _, _, r = x
            return r

        return sorted(self.mem, key=_s, reverse=True)[:k]
        
    

class Collector:
    def __init__(self) -> None:
        self.states = []
        self.actions = []
        self.rewards = []

        self.cur_states = []
        self.cur_actions = []

    def __len__(self):
        return len(self.states)

    def begin_record(self):
        self.cur_states = []
        self.cur_actions = []

    def add(self, state, action):
        self.cur_states.append(state)
        self.cur_actions.append(action)        

    def stop_record(self, reward):
        
        self.states += self.cur_states
        self.actions += self.cur_actions

        self.rewards += [reward for _ in range(len(self.cur_actions))]

        self.cur_actions = []
        self.cur_states = []


    @staticmethod
    def load(path):
        collector = Collector()
        
        file = h5py.File(path, 'r')

        collector.states = file['states'][:].copy()
        collector.actions = file['actions'][:].copy()
        collector.rewards = file['rewards'][:].copy()

        file.close()

        return collector

    def serialize(self, path):
        file = h5py.File(path, 'w')

        file.create_dataset('states', data=np.asarray(self.states))
        file.create_dataset('actions', data=np.array(self.actions))
        file.create_dataset('rewards', data=np.array(self.rewards))

        file.close()