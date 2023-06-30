from multiprocessing import Process, Queue
from utils.model import Model
from utils.game import Game
from numba import jit, cuda
from tqdm import tqdm
import threading, hashlib, random, numpy as np, math, copy, time, json

class Main:
    def __init__(self):
        file = json.load(open('model-training-data.py', 'r+'))

        self.brain = [np.array(data) for data in file[:2]] + file[2:]
        self.play()

    def render(self, grid):
        print('\n'.join([''.join([str(tile) for tile in row]) for row in grid]).replace('0', '-').replace('1', 'x').replace('2', 'o'))

    def play(self):
        model, heights, hidden_activation_function, output_activation_function, cost_function = self.brain
        model = Model(
            model=model,
            heights=heights,
            hidden_function = hidden_activation_function,
            output_function = output_activation_function,
            cost_function=cost_function
        )

        while True:
            game = Game()
            self.render(game.grid)
            points = 0
            while True:
                
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
                
                    grid = game.grid.flatten()
                    # rotation = game.normalization_rotation_offset
                    # grid = np.rot90(grid, rotation).flatten()

                    open_slots = game.open_slots.tolist()

                    real_grid = np.zeros(18)
                    real_grid[:9][grid == player + 1] = 1
                    real_grid[9:][grid == (1 - player) + 1] = 1

                    model_outputs = model.eval(
                        real_grid
                    )

                    model_outputs = [np.array(array) for array in model_outputs]

                    print(model_outputs[-1])

                    raw_prediction = np.array(model_outputs[-1])

                    mask = np.ones(raw_prediction.shape, bool)
                    mask[open_slots] = False

                    prediction = np.array(raw_prediction)
                    prediction[mask] = -1

                    choice = np.argmax(prediction)

                    best_moves = game.best_moves(1)

                    evaluation = game.select(choice, 1)

                    expected_output = np.array([0] * 9)
                    expected_output[best_moves] = 1

                    gradient, average_cost = model.gradient(
                        model_outputs, 
                        expected_output
                    )

                    print(real_grid)
                    print(raw_prediction, choice)
                    print(expected_output, best_moves)
                    print(average_cost)

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