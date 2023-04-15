from multiprocessing import Process, Queue
from numba import jit, cuda
from tqdm import tqdm
import threading, hashlib, random, numpy as np, math, copy, time, json

class Game:
    def __init__(self, grid=None):
        self.grid = np.array([ 
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]) if grid is None else np.array(np.split(np.array(grid), 3))

    @property
    def open_slots(self):
        grid = self.grid.flatten()
        return np.where(grid == 0)[0]

    @property
    def drawed(self):
        grid = self.grid
        threats = self._threats(grid, 0, 1) + self._threats(grid, 1, 1) + self._threats(grid, 0, 2) + self._threats(grid, 1, 2)

        return not bool(threats)

    @property
    def valid_game(self):

        if abs(len(np.where(self.grid == 1)[0]) - len(np.where(self.grid == 2)[0])) > 1:
            return False

        elif self._threats(self.grid, 0, 0) and self._threats(self.grid, 1, 0):
            return False
        
        return True

    def eval(self):
        for player in [1, 2]:
            if any([ 
                (self.grid[row] == player).all() for row in range(3)
            ]): # Horizontal crosses
                return player

            elif any([ 
                (self.grid[:, column] == player).all() for column in range(3)
            ]): # Vertical crosses
                return player

            elif (np.diag(self.grid) == player).all(): # Downwards diagonal crosses
                return player

            elif (np.diag(np.fliplr(self.grid)) == player).all() :# Upwards diagonal crosses
                return player

            elif not (self.grid == 0).any():
                return True
        
        return None

    def _threats(self, grid, player, tiles):

        search = ([0] * (3 - tiles)) + [player] * tiles
        
        rows = [[x * 3 + tile for tile in np.where(grid[x, :] == 0)[0]] for x in range(3) if sorted(grid[x, :]) == search]
        cols = [[tile * 3 + y for tile in np.where(grid[:, y] == 0)[0]] for y in range(3) if sorted(grid[:, y]) == search]

        diag = np.diag(grid)
        flip = np.diag(np.fliplr(grid))

        if sorted(diag) == search:
            diag1 =  [tile * 3 + tile for tile in np.where(diag == 0)[0]]
        else:
            diag1 = []
        
        if sorted(flip) == search:
            diag2 = [(2 - tile) * 3 + tile for tile in np.where(flip == 0)[0]]
        else:
            diag2 = []

        threats = sum(rows, []) + sum(cols, []) + diag1 + diag2

        return threats

    def select(self, tile, player):
        index = int(tile)

        opponent = (1 - int(player)) + 1
        player = int(player) + 1
        
        grid = self.grid

        x = index // 3
        y = index % 3

        win_threats =  self._threats(grid, player, 2)
        response = self.eval()

        if response is True: # If game id drawed
            return None

        if tile in win_threats: # If bot wins the game
            return True

        self.grid[x, y] = player

        return 1

    # Find the optimal moves, doesn't use mini-max
    def best_moves(self, player): 
        grid = self.grid

        opponent = (1 - int(player)) + 1
        player = int(player) + 1

        lose_threats = self._threats(grid, opponent, 2)
        win_threats =  self._threats(grid, player, 2)

        opp_forks = self._threats(grid, opponent, 1)
        forks = self._threats(grid, player, 1)

        real_forks = [threat for threat in [*set(opp_forks)] if opp_forks.count(threat) > 1]
        opp_forks = [threat for threat in [*set(opp_forks)] if opp_forks.count(threat) > 1 and ((threat in win_threats) if win_threats else True) ]
        forks = [threat for threat in [*set(forks)] if forks.count(threat) > 1 and ((threat in lose_threats) if lose_threats else True)]

        best = self.open_slots

        if win_threats:
            best = win_threats

        elif lose_threats:
            best = lose_threats

        elif len(real_forks) > 1: # To defend multiple fork possibilities you need create a threat that blocks a fork
            found = False

            for tile in self.open_slots:
                grid = np.array(list(self.grid))

                old_opp_forks = len(real_forks)
                old_win_threats = len(win_threats)
                grid[tile // 3, tile % 3] = player

                new_opp_forks = self._threats(grid, opponent, 1)
                new_win_threats = self._threats(grid, player, 2)

                new_opp_forks = [*set([threat for threat in new_opp_forks if new_opp_forks.count(threat) > 1])]
                
                if len(new_opp_forks) < old_opp_forks and not found:
                    best = [tile]

                if (len(new_opp_forks) < old_opp_forks) and new_win_threats and (not new_win_threats[0] in new_opp_forks):
                    best.append(tile)
                    found = True

        elif opp_forks:
            best = opp_forks

        elif 4 in self.open_slots:
            best = [4]

        elif (np.unique(self.grid).size == 2 and (opponent in self.grid)):
            opening = (np.where(self.grid == opponent)[0] * 3) + np.where(self.grid == opponent)[1]
            
            if (opening == 4):
                best = [0, 2, 6, 8]

            elif opening in [0, 2, 6, 8]:
                best = [4]

        return np.unique(best)

            
class Model:
    def __init__(self, model):
        self.model = model
        self._layer_outputs = []
        self._previous_changes = np.array([])
        np.seterr(all="ignore")

    # The activation function
    def _sigmoid(self, x):
        if x > 100:
            return 1

        return 1 / ( 1 + np.exp(-x) )

    # Run the model to get a output
    def eval(self, input):
        answer = 0
        model = self.model

        previous_layer_output = input

        self._layer_outputs = np.array([input])
        i = 0

        for layer in model:
            i += 1

            layer_output = np.array([])

            for node in layer:
                weights = node[:len(previous_layer_output)]
                bias = node[-1]

                output = self._sigmoid(np.dot(weights, previous_layer_output) + bias)

                layer_output = np.append(layer_output, output)

            self._layer_outputs = np.vstack([self._layer_outputs, layer_output])

            print(layer_output, i)

            previous_layer_output = np.array(layer_output)

        return layer_output

class Main:
    def __init__(self):
        self.brain = json.load(open('data.py', 'r+'))
        self.play()

    def render(self, grid):
        print('\n'.join([''.join([str(tile) for tile in row]) for row in grid]).replace('0', '-').replace('1', 'x').replace('2', 'o'))

    def play(self):
        while True:
            game = Game()
            self.render(game.grid)
            points = 0
            while True:

                model = Model(self.brain)
                

                print('Enter grid:')
                tile = int(input('>>')) - 1

                evaluation = game.select(tile % 9, 0)
                self.render(game.grid)

                if evaluation is True:
                    print("You win!")
                    break

                elif game.drawed:
                    print("Game drawed!")
                    break

                points += evaluation
                    

                
                grid = game.grid.flatten()
                open_slots = game.open_slots

                prediction = model.eval(grid)
                largest_value = np.max(prediction[open_slots])
                choice = np.where(prediction == largest_value)[0][0]

                print(prediction)

                evaluation = game.select(choice, 1)

                print("+++ Bot's Turn")

                self.render(game.grid)

                if evaluation is True:
                    print("You lose!")
                    break

                elif game.drawed:
                    print("Game drawed!")
                    break



if __name__ == '__main__':
    Main()