import tkinter as tk
import numpy as np


class GamePanel:
    CELL_PADDING = 10
    BACKGROUND_COLOR = '#92877d'
    EMPTY_CELL_COLOR = '#9e948a'
    CELL_BACKGROUND_COLOR_DICT = {
        '2': '#eee4da',
        '4': '#ede0c8',
        '8': '#f2b179',
        '16': '#f59563',
        '32': '#f67c5f',
        '64': '#f65e3b',
        '128': '#edcf72',
        '256': '#edcc61',
        '512': '#edc850',
        '1024': '#edc53f',
        '2048': '#edc22e',
        'beyond': '#3c3a32'
    }
    CELL_COLOR_DICT = {
        '2': '#776e65',
        '4': '#776e65',
        '8': '#f9f6f2',
        '16': '#f9f6f2',
        '32': '#f9f6f2',
        '64': '#f9f6f2',
        '128': '#f9f6f2',
        '256': '#f9f6f2',
        '512': '#f9f6f2',
        '1024': '#f9f6f2',
        '2048': '#f9f6f2',
        'beyond': '#f9f6f2'
    }
    FONT = ('Verdana', 24, 'bold')
    UP_KEYS = ('w', 'W', 'Up')
    LEFT_KEYS = ('a', 'A', 'Left')
    DOWN_KEYS = ('s', 'S', 'Down')
    RIGHT_KEYS = ('d', 'D', 'Right')

    def __init__(self, grid: np.ndarray, moves):
        self.grids = grid.astype(int)
        self.moves = moves


        self.cur = 0
        self.root = tk.Tk()
        self.root.resizable(True, True)
        
        self.move_text = tk.Label(text='2048')
        self.move_text.pack()

        self.background = tk.Frame(self.root, bg=GamePanel.BACKGROUND_COLOR)
        self.cell_labels = []

        self.background.pack(side=tk.TOP)
        

        self.root.bind("n", self._next)
        self._next(None)

    def _next(self, e):
        if self.cur + 1 > len(self.grids):
            print('end')
            return
        
        grid = self.grids[self.cur]

        self._paint(grid)
        self.cur += 1

    def _paint(self, grid: np.ndarray):
        cell_labels = []
        for i in range(grid.shape[0]):
            row_labels = []
            for j in range(grid.shape[1]):
                label = tk.Label(self.background, text='',
                                bg=GamePanel.EMPTY_CELL_COLOR,
                                justify=tk.CENTER, font=GamePanel.FONT,
                                width=4, height=2)
                label.grid(row=i, column=j, padx=10, pady=10)
                row_labels.append(label)
            cell_labels.append(row_labels)

        for i in range(grid.shape[0]):
            for j in range(grid.shape[0]):
                if grid[i, j] == 1:
                    cell_labels[i][j].configure(
                         text='',
                         bg=GamePanel.EMPTY_CELL_COLOR)
                else:
                    cell_text = str(grid[i, j])

                    bg_color = GamePanel.CELL_BACKGROUND_COLOR_DICT.get(cell_text)
                    fg_color = GamePanel.CELL_COLOR_DICT.get(cell_text)

                    cell_labels[i][j].configure(
                        text=cell_text,
                        bg=bg_color, fg=fg_color)

if __name__ == '__main__':
    import torch
    from game.state import State
    from game.enums import Direction
    from model import model, device

    model.load_state_dict(torch.load('weights7.pt'))
    model.eval()

    game = State().random_spawn()

    s = []
    actions = []

    cnt = 1
    while game.can_play():
        print(f"move #{cnt}")
        cnt += 1

        state = game.to_numpy()
        s.append(state)
        state = torch.FloatTensor(state.copy()).unsqueeze(0).unsqueeze(0).to(device)
        
        condidates = torch.multinomial(model(state), num_samples=4, replacement=False)

        for action in condidates[0]:
            if action.item() in game.get_valid_moves():
                game = game.apply_move(Direction(action.item()))
                actions.append(str(Direction(action.item())))
                break
        else:
            print(game.to_numpy())
            raise Exception('no legal moves')
        
        if game.can_play():
            game = game.random_spawn()

        if cnt % 1000 == 0:
            print(game.to_numpy())
    
    s.append(game.to_numpy())

    print("Win" if game.is_win() else "Lose")    

    panel = GamePanel(np.asarray(s), actions)
    panel.root.mainloop()
    
