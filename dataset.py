import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

from torchvision import transforms

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class GameDataset(Dataset):
    def __init__(self, data_files):
        self.files = data_files    

        self.length = 0

        self.state = []
        self.action = []
        self.reward = []

        for name in self.files:
            f = h5py.File(name, 'r')

            self.length += len(f['states'][:])
        
            self.state.append(f['states'][:].copy())
            self.action.append(f['actions'][:].copy())
            self.reward.append(f['rewards'][:].copy())

            f.close()      

        self.state = np.concatenate(self.state).astype(np.float32)
        self.action = np.concatenate(self.action)
        self.reward = np.concatenate(self.reward).astype(np.float32)

        print(self.state.shape)
        
    def __getitem__(self, idx):
        s = torch.from_numpy(self.state[idx]).unsqueeze(0)
        c = torch.zeros_like(s)
        c[s != 0] = 1 / s[s != 0]
    
        return (c.to(device), 
                torch.as_tensor(self.action[idx], dtype=torch.long).to(device), 
                torch.as_tensor(self.reward[idx], dtype=torch.float).to(device))

    def __len__(self):
        return self.length
    
class ExpirienceDataset(Dataset):
    def __init__(self, state, action, reward) -> None:
        super().__init__()

        self.state = torch.cat(state)
        self.action = torch.tensor(action, dtype=torch.long)
        self.reward = torch.tensor(reward, dtype=torch.float)

    def __getitem__(self, idx):
        s = self.state[idx]
        c = torch.zeros_like(s)
        c[s != 0] = 1 / s[s != 0]

        s = c

        return (c.to(device), self.action[idx].to(device), self.reward[idx].to(device))

    def __len__(self):
        return len(self.state)
        