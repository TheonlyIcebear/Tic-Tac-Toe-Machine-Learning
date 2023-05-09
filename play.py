from multiprocessing import Process, Queue
from numba import jit, cuda
from utils.game import Game
from tqdm import tqdm
import threading, hashlib, random, numpy as np, math, copy, time, json
            
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
    @jit(forceobj=True)
    def eval(self, input):
        answer = 0
        model = self.model

        previous_layer_output = input

        self._layer_outputs = np.array([])

        for count, layer in enumerate(model):

            layer_output = np.array([])
 
            for node in layer:
                weights = node[:len(previous_layer_output)]
                bias = node[-1]

                output = self._sigmoid(np.dot(weights, previous_layer_output) + bias)

                layer_output = np.append(layer_output, output)


            if self._layer_outputs.size:
                self._layer_outputs = np.vstack([self._layer_outputs, layer_output])
            else:
                self._layer_outputs = np.hstack([self._layer_outputs, layer_output])

            previous_layer_output = np.array(layer_output)

        return layer_output[:9]

class Main:
    def __init__(self):
        self.brain = json.load(open('model-training-data.py', 'r+'))
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
                if (game.grid == 0).all():
                    player = random.randint(0, 1)

                else:
                    player = 1 - player
                
                if not player:

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
                    
                else:
                
                    grid = np.array(game.grid.flatten())
                    open_slots = game.open_slots

                    real_grid = grid
                    real_grid[grid == 2] = -1

                    prediction = model.eval(real_grid)
                    largest_value = np.max(prediction[open_slots])
                    choice = np.where(prediction == largest_value)[0][0]

                    print(prediction, game.best_moves(1))

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