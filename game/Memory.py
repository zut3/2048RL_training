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
        self.cur_rewards = []
    
    def __len__(self):
        return len(self.states)
    
    def begin_record(self):
        self.cur_states = []
        self.cur_actions = []
        self.cur_rewards = []
    
    def add(self, state, action, reward=None):
        self.cur_states.append(state)
        self.cur_actions.append(action) 
        if reward is not None:
            self.cur_rewards.append(reward)

    def stop_record(self, reward=None):
        self.states += self.cur_states
        self.actions += self.cur_actions

        if reward is None:
            self.rewards += self.cur_rewards
        else:
            self.rewards += [reward for _ in self.cur_actions]

    def topk(self, k=100):       
        zipped = list(zip(self.states, self.actions, self.rewards))
        
        def _f(x):
            _, _, r = x
            return r

        res = sorted(zipped, key=_f, reverse=True)[:k]
        
        states = [z[0] for z in res]
        actions = [z[1] for z in res]
        rewards = [z[1] for z in res]

        return states, actions, rewards


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