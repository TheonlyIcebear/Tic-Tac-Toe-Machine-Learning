from multiprocessing import Process, Queue
from numba import jit, cuda
from operations import *
from tqdm import tqdm
import threading, hashlib, random, numpy as np, math, copy, time, json

class Game:
    def __init__(self):
        self.grid = np.array([ 
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])

    def threats(self, grid, player, tiles):

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


        lose_threats = self.threats(grid, opponent, 2)
        win_threats =  self.threats(grid, player, 2)

        opp_forks = self.threats(grid, opponent, 1)
        forks = self.threats(grid, player, 1)

        real_forks = [threat for threat in [*set(opp_forks)] if opp_forks.count(threat) > 1]
        opp_forks = [threat for threat in [*set(opp_forks)] if opp_forks.count(threat) > 1 and ((threat in win_threats) if win_threats else True) ]
        forks = [threat for threat in [*set(forks)] if forks.count(threat) > 1 and ((threat in lose_threats) if lose_threats else True)]

        points = 0

        response = self.eval()

        if response is True:
            return None

        if win_threats:
            if not tile in win_threats: # If bot ignores a free win
                points -= 2

            else:
                points += 2

        elif lose_threats:
            if not tile in lose_threats: # If bot hangs a loss
                points -= 1
            else:
                points += 0.5

        elif len(real_forks) > 1: # To defend multiple fork possibilities you need create a threat that blocks a fork

            old_opp_forks = len(real_forks)
            old_win_threats = len(win_threats)
            grid[x, y] = player

            new_opp_forks = self.threats(grid, opponent, 1)
            new_win_threats = [threat for threat in self.threats(grid, player, 2) if not threat in opp_forks]

            new_opp_forks = [*set(threat for threat in new_opp_forks if new_opp_forks.count(threat) > 1)]

            if new_win_threats:
                points += 5
            else:
                points -= 5

        elif opp_forks:
            if not tile in opp_forks: # Bot fails to block fork threat
                points -= 1

            else:
                points += 0.5

        elif (np.unique(self.grid).size == 2 and (opponent in self.grid)):
            opening = np.where(self.grid == opponent)[0] * 3 + np.where(self.grid == opponent)[1]
            
            if (opening == 4):
                if tile in [0, 2, 6, 8]:
                    points += 75

                else:
                    points -= 10000

            elif opening in [0, 2, 6, 8]:
                if tile == 4:
                    points += 75

                else:
                    points -= 10000

        elif tile == 4:
            points += 0.25

        elif tile in [0, 2, 6, 8]:
            points += 0.125

        self.grid[x, y] = player

        return points

    @property
    def open_slots(self):
        grid = self.grid.flatten()
        return np.where(grid == 0)[0]

    @property
    def drawed(self):
        grid = self.grid
        threats = self.threats(grid, 0, 1) + self.threats(grid, 1, 1) + self.threats(grid, 0, 2) + self.threats(grid, 1, 2)

        return not bool(threats)

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

    def random_move(self, player, skill_level):
        grid = self.grid

        if not skill_level:
            return random.choice(self.open_slots)

        opponent = (1 - int(player)) + 1
        player = int(player) + 1

        lose_threats = self.threats(grid, opponent, 2)
        win_threats =  self.threats(grid, player, 2)

        opp_forks = self.threats(grid, opponent, 1)
        forks = self.threats(grid, player, 1)

        real_forks = [threat for threat in [*set(opp_forks)] if opp_forks.count(threat) > 1]
        opp_forks = [threat for threat in [*set(opp_forks)] if opp_forks.count(threat) > 1 and ((threat in win_threats) if win_threats else True) ]
        forks = [threat for threat in [*set(forks)] if forks.count(threat) > 1 and ((threat in lose_threats) if lose_threats else True)]

        points = 0

        if win_threats:
            best = random.choice(win_threats)

        elif lose_threats:
            best = random.choice(lose_threats)

        elif len(real_forks) > 1: # To defend multiple fork possibilities you need create a threat that blocks a fork
            for tile in self.threats(grid, player, 1):
                grid = np.array(list(self.grid))

                old_opp_forks = len(real_forks)
                old_win_threats = len(win_threats)
                grid[tile // 3, tile % 3] = player

                new_opp_forks = self.threats(grid, opponent, 1)
                new_win_threats = self.threats(grid, player, 2)

                new_opp_forks = [*set([threat for threat in new_opp_forks if new_opp_forks.count(threat) > 1])]
                
                if len(new_opp_forks) < old_opp_forks:
                    best = tile
                    
                if (len(new_opp_forks) < old_opp_forks) and len(new_win_threats) > old_win_threats:
                    best = tile
                    break

        elif opp_forks:
            best = random.choice(opp_forks)

        elif 4 in self.open_slots:
            best =  4

        elif np.unique(self.grid).tolist() == [0, opponent] and self.grid[1][1] == opponent:
            best = random.choice([0, 2, 6, 8])

        else:
            best = random.choice(self.open_slots)

        decimal_accuracy = 256

        if random.randint(0, 10 ** decimal_accuracy // skill_level) <= 10 ** decimal_accuracy:
            return best
        
        else:
            return random.choice(self.open_slots)
            
class Model:
    def __init__(self, model):
        self.model = model
        self._layer_outputs = []

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

        for layer in model:

            layer_output = np.array([])

            for node in layer:
                weights = node[:len(previous_layer_output)]
                bias = node[-1]

                output = self._sigmoid(np.dot(weights, previous_layer_output) + bias)

                layer_output = np.append(layer_output, output)

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

                evaluation = game.select(tile % 9, 1)
                self.render(game.grid)

                if evaluation == 2:
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

                # choice = game.random_move(0, 1)
                evaluation = game.select(choice, 0)

                print(evaluation)

                print("+++ Bot's Turn")

                self.render(game.grid)

                if evaluation == 2:
                    print("You lose!")
                    break

                elif game.drawed:
                    print("Game drawed!")
                    break



if __name__ == '__main__':
    Main()