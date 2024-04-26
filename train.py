import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from game.state import State

from model import model
from game.Memory import Collector
from game.enums import Direction
from torch.utils.data import DataLoader
from dataset import GameDataset, ExpirienceDataset


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def get_stats(model, num, temperature=0) -> Collector:
    model.eval()

    collector = Collector()

    wins = 0

    for i in range(num):
        print(f'starting new game #{i+1}')

        game = State().random_spawn()

        collector.begin_record()

        while game.can_play():
            state = torch.FloatTensor(game.to_numpy().copy()).unsqueeze_(0).unsqueeze_(0).to(device)
            
            pred = model(state).detach().cpu()
            condidates = torch.multinomial(pred, num_samples=4, replacement=False)
            
            valid = game.get_valid_moves()

            for action in condidates[0]:
                if Direction(action.item()) in valid:
                    break
            else:
                raise Exception('no legal moves')
            
            action = action.item()

            if np.random.random() < temperature:
                action = int(np.random.choice(valid))
            
            game = game.apply_move(action)
            collector.add(state.cpu(), action)

            if game.can_play():
                game = game.random_spawn()

            torch.cuda.empty_cache()
        
        collector.stop_record(1 if game.is_win() else -1)

        if game.is_win():
            print('Win!!\n')
            wins += 1

    return collector, wins

def eval(model, val_dataloader, criterion=torch.nn.CrossEntropyLoss()):
    model.eval()

    avg_loss = 0
    acc = 0 

    for s, a, _  in val_dataloader:
        pred = model(s)
        acc += torch.mean((torch.argmax(pred, dim=1) == a).float())

        loss = criterion(pred, a)

        avg_loss += loss.detach().cpu().item()

    return avg_loss / len(val_dataloader), acc.cpu().item() / len(val_dataloader)

def train_on_data(data_files, model, opt, epochs=1, batch_size=64, val_frac=0.2, weights_file='weights.pt', scheduler=None):
    history = {
        "loss": [],
        "val_loss": [],
    }

    criterion = torch.nn.CrossEntropyLoss()
    
    ds = GameDataset(data_files)

    val_size = int(len(ds) * val_frac)
    train_set, val_set = torch.utils.data.random_split(ds, [len(ds) - val_size, val_size])

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    _, acc = eval(model, val_dataloader)
    print('model accuracy before training: ', round(acc, 7))

    for e in range(1, epochs+1):
        
        avg_loss = 0
        model.train()

        for s, a, r in train_dataloader:            
            opt.zero_grad()
            pred = model(s)
            
            loss = criterion(pred, a)
            loss.backward()
            opt.step()
            
            avg_loss += loss.detach().cpu().item()

        avg_loss /= len(train_dataloader)
        
        history['loss'].append(avg_loss)
        val_loss, acc = eval(model, val_dataloader)
        history['val_loss'].append(val_loss)

        if scheduler is not None:
            scheduler.step()

        torch.cuda.empty_cache()

        print(f"epoch #{e} avg loss {round(avg_loss, 5)} accuracy {round(acc, 5)}")

    torch.save(model.state_dict(), weights_file)

    return history

def train_policy_gradient(model, opt, epochs, val_dataloader, weights_file='weights.pt', num_games=5):
    history = {
        'loss': [],
        'win_rate': []
    }
    
    criterion = torch.nn.CrossEntropyLoss()

    cur_temp = 0.7
    max_win_rate = 0

    for e in range(1, epochs+1):
        collector, wins = get_stats(model, num_games, temperature=cur_temp)

        rewards = torch.zeros((len(collector.rewards), 4))

        for i, r in enumerate(collector.rewards):
            rewards[i][collector.actions[i]] = r

        ds = ExpirienceDataset(collector.states, collector.actions, rewards)
        dataloader = DataLoader(ds, shuffle=True, batch_size=1024)

        avg_loss = 0
        model.train()

        for s, _, r in dataloader:            
            opt.zero_grad()
            pred = model(s)
            
            loss = criterion(pred, r)
            loss.backward()
            opt.step()
            
            avg_loss += loss.detach().cpu().item()

        win_rate = wins / num_games

        if win_rate >= max_win_rate and win_rate != 0:
            max_win_rate = win_rate
            cur_temp -= 0.05
            cur_temp = max(cur_temp, 0.1)

        avg_loss /= len(dataloader)

        history['loss'].append(avg_loss)
        history['win_rate'].append(wins / num_games)

        if e % 10 == 0:
            val_loss, acc = eval(model, val_dataloader)
            print(f"Accuracy on epoch #{e}", acc)

        print(f"Epoch #{e} avg loss {avg_loss}")

    torch.save(model.state_dict(), weights_file)

    return history

if __name__ == '__main__':  
    import os

    #model.load_state_dict(torch.load('weights3.pt'))
    
    # opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-6)
    # scheduler = StepLR(opt, 100, 0.5)

    # files = os.listdir('./data')
    # files = [os.path.join('./data', f) for f in files] 

    # print(files)

    # hist = train_on_data(files, model, opt, 100, batch_size=1024, weights_file='weights5.pt', scheduler=scheduler)
    
    # plt.plot(hist['loss'], label='train loss')
    # plt.plot(hist['val_loss'], label='validation loss')
    # plt.legend(loc="upper right")

    # plt.show()

    # ds = GameDataset(data_files=['./data/games_1.h5', './data/games_2.h5'])
    # val_dataloader = DataLoader(ds, batch_size=1024, shuffle=False)

    #model.load_state_dict(torch.load('weights5.pt'))

    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    
    ds = GameDataset(data_files=['./data/games_1.h5', './data/games_2.h5'])
    val_dataloader = DataLoader(ds, batch_size=1024, shuffle=False)

    hist = train_policy_gradient(model, opt, 50, val_dataloader, weights_file='weights7.pt', num_games=7)
    val_loss, acc = eval(model, val_dataloader)

    print("Accuracy: ", acc)

    fig, ax = plt.subplots(2, 1)

    ax[0].plot(hist['loss'], label='loss')
    ax[0].legend(loc="upper right")

    ax[1].plot(hist['win_rate'], label='win_rate', c='y')
    ax[1].legend(loc="upper right")

    plt.show()

    