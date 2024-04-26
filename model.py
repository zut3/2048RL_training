from torch import nn
import torch

if torch.cuda.is_available:
    device = 'cuda'
else:
    device = 'cpu'

class Block(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        
        self.relu = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(inc, outc, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(outc, outc, kernel_size=2, stride=2, padding=3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv3(x)

        return x

model = nn.Sequential(
    Block(1, 4),
    nn.ELU(),
    Block(4, 8),
    nn.BatchNorm2d(8),    
    nn.ELU(),
    nn.Flatten(),
    
    nn.Linear(5*5*8, 1024),
    nn.ELU(),
    nn.Dropout(0.25),
    nn.Linear(1024, 512),
    nn.ELU(),
    nn.Linear(512, 4),

    nn.Softmax(dim=-1)
).to(device)
    
class QFunction(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.state_proc = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=2, stride=2, padding=3),
            nn.ReLU(),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=2, stride=2, padding=3),
            nn.ReLU(),

            nn.Flatten(),
            nn.Dropout(0.3),
        )
 
        self.fc = nn.Sequential(
            nn.Linear(5*5*16 + 4, 512),
            nn.ELU(),
            nn.Linear(512, 1)
        )
        self.tanh = nn.Tanh()

    def forward(self, state, action):
        action = torch.nn.functional.one_hot(action, 4)
        state_vec = self.state_proc(state)
        
        x = torch.cat([state_vec, action], dim=-1)
        
        x = self.fc(x)
        x = self.tanh(x)

        return x