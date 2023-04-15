from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
from numba import jit, cuda
from tqdm import tqdm
import multiprocessing, threading, hashlib, random, numpy as np, copy, math, time, json

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

    # Edit the model according to the gradients
    def apply_changes(self, gradients, learning_rate, rev=1):
        model = np.array(self.model)
        height = model.shape[1]

        for gradient in gradients:
            for mask in gradient:
                model += (mask * learning_rate) * rev

        self.model = model

    # Using back propagration to calculate the gradient descent
    def gradient(self, target, predictions, momentum=0.5):

        model = np.array(self.model)

        height = self.model.shape[1]
        length = self.model.shape[0]

        last_layer = predictions

        targets = np.array([0] * height)
        targets[target] = 1

        layer_outputs = self._layer_outputs

        model_copy = np.array(model, dtype=float)
        model_copy[:, :, :] = 0.

        _previous_derivs = np.array([])
        
        for count, (outputs, layer) in enumerate(zip(layer_outputs[:0:-1], model[::-1])):
            
            previous_derivs = np.array(_previous_derivs)
            _previous_derivs = np.array([])

            if count == length - 1:
                break

            for idx, (output, weights) in enumerate(zip(outputs, layer)):
                
                prediction = layer_outputs[count + 1][idx]

                weights = weights[:-1]
                bias = weights[-1]
                sigmoid_deriv = prediction * (1 - prediction)

                if self._previous_changes.size:
                    momentum_velocity = self._previous_changes[count, idx] * momentum
                else:
                    momentum_velocity = 0


                prev_activations = layer_outputs[count + 1]

                if not count:

                    target = targets[idx]

                    cost_deriv = 2 * (prediction - target)
                    node_value = cost_deriv * sigmoid_deriv

                    total_deriv = prev_activations * node_value
                    bias_deriv = node_value

                    # model_copy[-(count + 1), idx, :] -= np.append(total_deriv, bias_deriv) + momentum_velocity


                    _previous_derivs = np.append(_previous_derivs, node_value)

                    continue
                
                node_value = sigmoid_deriv * np.dot(weights, previous_derivs)

                bias_deriv = node_value
                total_deriv = prev_activations * node_value

                model_copy[-(count + 1), idx, :] -= np.append(total_deriv, bias_deriv) + momentum_velocity

                _previous_derivs = np.append(_previous_derivs, node_value)

        self._previous_changes = model_copy

        return model_copy

    # Run the model to get a output
    def eval(self, input):
        answer = 0
        model = self.model

        previous_layer_output = input

        self._layer_outputs = np.array([input])

        for layer in model:

            layer_output = np.array([])

            for node in layer:
                weights = node[:len(previous_layer_output)]
                bias = node[-1]

                output = self._sigmoid(np.dot(weights, previous_layer_output) + bias)

                layer_output = np.append(layer_output, output)

            self._layer_outputs = np.vstack([self._layer_outputs, layer_output])

            previous_layer_output = np.array(layer_output)

        return layer_output

class Main:
    def __init__(self, tests, learning_rate, momentum_conservation, dimensions, threads):
        self.tests = tests
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
            brain = np.array(json.load(open('data.py', 'r+')))
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

            if generation:
                best_overtime[generation - 1] = self.accuracy

            json.dump(best_overtime, open('generations.json', 'w+'), indent=2)

    # Manages threads
    def manager(self, model=None):
        threads = []
        average_accuracy = np.array([])

        queue = self.queue
        
        for count in range(self.threads):
            receive_queue = Queue()

            thread = multiprocessing.Process(target=self.worker, args=(receive_queue,))
            threads.append([thread, receive_queue])

        for thread in threads:
            thread[0].start()


        count = 0
        old_accuracy = -1
        model = Model(model)

        while True:
            
            count += 1
            self.generations += 1 
            self.model = model

            self.children = self.threads
            self.accuracy = 0
            
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

            average_accuracy = np.append(average_accuracy, self.accuracy)

            backup = open('data.py', 'r+').read()
            queue = self.queue

            if self.accuracy >= old_accuracy:

                old_accuracy = float(self.accuracy)

                with open('data.py', 'w+') as file:

                        try:
                            file.write(json.dumps(model.model.tolist()))
                        except Exception as e:
                            print(e)
                            file.write(backup)

    def worker(self, receieve_queue=None):
        generations = 0

        while True:
            generations += 1
            brain = receieve_queue.get()

            gradient_map = []
            points = 0

            for count in range(self.tests):
                player = count % 2
                model = brain
                game = Game()

                for i in range(9):

                    player = 1 - player
                    grid = game.grid.flatten()

                    open_slots = game.open_slots.tolist()

                    if not open_slots:
                        break

                    if player == 1:
                        
                        prediction = model.eval(grid)
                        largest_value = np.max(prediction[open_slots])
                        choice = np.where(prediction == largest_value)[0][0]

                        best_moves = game.best_moves(player)

                        gradient = model.gradient(best_moves, prediction)
                        gradient_map.append(gradient)

                        select = game.select(choice, player)

                        if choice in best_moves:
                            points += 1

                        if game.drawed:
                            break

                        if select is True:
                            break

                    else:
                        if not np.random.randint(2):
                            move = random.choice(game.open_slots)
                        else:
                            move = random.choice(game.best_moves(player))

                        select = game.select(move, player)

                        if game.drawed:
                            break

                        if select is True:
                            break

                accuracy = points/self.tests
                self.queue.put([accuracy, gradient_map])
                    

    def build(self):

        range = np.delete(np.arange(-1000, 1000), 1000)
        model = 1 / np.random.choice(range, size=(self.length, self.height, self.inputs + 1))

        return model

if __name__ == "__main__":
    Main(
        tests = 50,  # The length of the tests,
        momentum_conservation = 0.75, # What percent of the previous changes that are added to each weight in our gradient descent
        learning_rate = 0.75, # How fast the model learns, if too low the model will train very slow and if too high it won't train
        dimensions = [350, 9],  # The dimensions of the model
        threads = 10  # How many concurrent threads to be used
    )