from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
from utils.game import Game
from numba import jit, cuda
from tqdm import tqdm
import multiprocessing, threading, hashlib, random, numpy as np, copy, math, time, json
            
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

    # Edit the model according to the gradients
    def apply_changes(self, gradients, learning_rate):
        model = np.array(self.model)
        height = model.shape[1]

        for gradient in gradients:
            for count, mask in enumerate(gradient):
                model -= (mask * learning_rate)

        self.model = model

    # Using back propagration to calculate the gradient descent
    def gradient(self, target, predictions, momentum=0.5):

        model = np.array(self.model)

        length = self.model.shape[0]

        input_layer = predictions

        targets = np.array([0] * 9)
        targets[target] = 1

        layer_outputs = self._layer_outputs[::-1]

        model_copy = np.array(model, dtype=float)
        model_copy[:, :, :] = 0.

        _previous_derivs = np.array([])

        self.average_cost = ((predictions - targets) ** 2).mean()
        
        for count, (outputs, layer) in enumerate(zip(layer_outputs, model[::-1])):
            
            previous_derivs = np.array(_previous_derivs)
            _previous_derivs = np.array([])

            for idx, (output, weights) in enumerate(zip(outputs[::-1], layer[::-1])):

                if count == length - 1:
                    break

                elif (not count) and (idx >= 9):
                    continue

                else:
                    prev_activations = layer_outputs[count + 1]
                    prediction = output

                index = -(count + 1)
                layer_height = len(outputs)
                prev_layer_height = len(prev_activations)

                weights = weights[:layer_height]
                bias = weights[-1]
                sigmoid_deriv = prediction * (1 - prediction)

                if self._previous_changes.size:
                    momentum_velocity = self._previous_changes[count, idx, :prev_layer_height + 1] * momentum
                else:
                    momentum_velocity = 0

                if not count:

                    target = targets[idx]

                    cost_deriv = 2 * (prediction - target)
                    node_value = cost_deriv * sigmoid_deriv

                    total_deriv = prev_activations * node_value
                    bias_deriv = node_value

                    model_copy[index, idx, :layer_height + 1] = np.append(total_deriv, bias_deriv) + momentum_velocity

                    _previous_derivs = np.append(_previous_derivs, node_value)

                    continue
                
                node_value = sigmoid_deriv * np.dot(weights[:len(previous_derivs)], previous_derivs)

                bias_deriv = node_value
                total_deriv = prev_activations * node_value

                model_copy[index, idx, :layer_height + 1] = np.append(total_deriv, bias_deriv) + momentum_velocity

                _previous_derivs = np.append(_previous_derivs, node_value)

        self._previous_changes = model_copy

        return model_copy

    # Run the model to get a output
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
    def __init__(self, tests_amount, generation_limit, learning_rate, momentum_conservation, dimensions, threads):
        self.tests = tests_amount
        self.trials = generation_limit
        self.learning_rate = learning_rate
        self.momentum = momentum_conservation
        self.length, self.height = dimensions
        self.threads = threads
        
        self.queue = Queue()
        self.children = 0
        self.generations = 0
        self.accuracy = 0
        self.model = []
        self.inputs = 9 # There are nine inputs for the model as there are nine tiles on a tic tac toe grid
        
        
        try:
            brain = np.array(json.load(open('model-training-data.py', 'r+')))
        except Exception as e:
            print(e)
            brain = self.build()

        threading.Thread(target=self.manager, args=(brain,)).start()
        self.update()

    def generator(self): 
        while True:
            yield

    def update(self):
        tloop = tqdm(self.generator())
        
        best_overtime = []
        
        for _ in tloop:

            tloop.set_description(f"Average Accuracy: {self.accuracy}, Generations: {self.generations}, Live: {self.children}")

            generation = int(self.generations)
            if not len(best_overtime) == generation:
                best_overtime.append(0)

            try:
                if generation:
                    best_overtime[generation - 1] = self.accuracy
            except:
                continue

            json.dump(best_overtime, open('generations.json', 'w+'), indent=2)

    # Manages threads
    def manager(self, model=None):
        threads = []
        average_accuracy = np.array([])

        queue = self.queue
        
        for count in range(self.threads):
            receive_queue = Queue()

            thread = multiprocessing.Process(target=self.worker, args=(receive_queue, count,))
            threads.append([thread, receive_queue])

        for thread in threads:
            thread[0].start()


        count = 0
        old_accuracy = -1
        model = Model(model)

        for _ in range(self.trials):
            
            count += 1
            self.generations += 1 
            self.model = model

            self.children = self.threads
            self.accuracy = 0

            backup = open('model-training-data.py', 'r+').read()
            queue = self.queue

            if self.accuracy >= old_accuracy:

                old_accuracy = float(self.accuracy)

                with open('model-training-data.py', 'w+') as file:

                        try:
                            file.write(json.dumps(model.model.tolist()))
                        except Exception as e:
                            print(e)
                            file.write(backup)
            
            for thread in threads:
                thread[1].put(model)

            gradients = []

            for _ in range(self.threads):
                accuracy, gradient_map = queue.get()
                gradients.append(gradient_map)
                
                self.accuracy = (self.accuracy + accuracy) / 2
                self.children -= 1

            model.apply_changes(gradients, self.learning_rate)
            # model = Model(model.model.tolist())

    def worker(self, receieve_queue=None, thread_index=0):
        generations = 0
        start = 0

        possible_games = []

        for count in range(3 ** 9):
            grid = [(count // (3 ** i)) % 3 for i in range(9)]

            game = Game(grid)
            if abs(grid.count(1) - grid.count(2)) > 1 or game.eval():
                continue

            possible_games.append(grid)

        while True:
            generations += 1
            brain = receieve_queue.get()

            gradient_map = []
            trials = 0
            points = 0

            for player in range(2):

                for count in range(start + self.tests * thread_index, start + self.tests * (thread_index + 1)):
                    model = brain
                    grid = possible_games[count % len(possible_games)]
                    
                    game = Game(grid)
                    open_slots = game.open_slots.tolist()

                    x_grid = np.zeros(9)
                    o_grid = np.zeros(9)
                    blank_grid = np.zeros(9)

                    blank_grid[np.where(grid == 0)] = 1
                    x_grid[np.where(grid == 1)] = 1
                    o_grid[np.where(grid == 2)] = 1
                    
                    real_grid = np.append(blank_grid, np.append(x_grid, np.append(o_grid, player)))
                        
                    prediction = model.eval(real_grid)
                    largest_value = np.max(prediction[open_slots])
                    choice = np.where(prediction == largest_value)[0][0]

                    best_moves = game.best_moves(player)

                    gradient = model.gradient(best_moves, prediction)
                    gradient_map.append(gradient)

                    select = game.select(choice, player)

                    if choice in best_moves:
                        points += model.average_cost

                    trials += 1
                        

            if not trials:
                accuracy = 0
            else:
                accuracy = points / trials
            self.queue.put([accuracy, gradient_map])

            start += self.tests * self.threads
                    

    def build(self):
        default = (3 * 9) + 2

        model = np.random.randn(self.length, self.height, self.height + 1 if self.height > default else default )

        return model

if __name__ == "__main__":
    Main(
        tests_amount = 150, # The length of the tests,
        generation_limit = 500, # The amount of generations the model will be trained through
        momentum_conservation = 0.00, # What percent of the previous changes that are added to each weight in our gradient descent
        learning_rate = 0.0025, # How fast the model learns, if too low the model will train very slow and if too high it won't train
        dimensions = [2, 36],  # The length and height of the model
        threads = 6  # How many concurrent threads to be used
    )