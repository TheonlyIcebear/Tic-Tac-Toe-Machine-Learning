from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
from utils.model import Model
from utils.game import Game
from numba import jit, cuda
from tqdm import tqdm
import multiprocessing, threading, hashlib, random, numpy as np, copy, math, time, json, os

class Main:
    def __init__(self, tests_amount, generation_limit, learning_rate, momentum_conservation, weight_decay, cost_limit, dimensions, threads):
        self.tests = tests_amount
        self.trials = generation_limit
        self.learning_rate = learning_rate
        self.momentum = momentum_conservation
        self.wd = weight_decay
        self.length, self.height = dimensions[0]
        self.shape = dimensions[1]
        self.threads = threads
        self.cost_limit = cost_limit
        
        self.queue = Queue()
        self.update_queue = Queue()
        self.children = 0
        self.generations = 0
        self.accuracy = 0
        self.model = []
        self.length += 1
        self.inputs = 9 # There are nine inputs for the model as there are nine tiles on a tic tac toe grid
        
        
        try:
            file = json.load(open('model-training-data.py', 'r+'))

            brain = [np.array(data) for data in file[:2]] + file[2:]
        except Exception as e:
            brain = self.build()

        threading.Thread(target=self.manager, args=(brain,)).start()
        self.update()

    def generator(self): 
        while True:
            yield

    def update(self):
        tloop = tqdm(self.generator())
        queue = self.update_queue
        
        cost_overtime = []
        
        for _ in tloop:

            tloop.set_description(f"Average Cost: {self.accuracy}, Generations: {self.generations}, Live: {self.children}")

            cost = queue.get()
            cost_overtime.append(cost)

            json.dump(cost_overtime, open('generations.json', 'w+'), indent=2)

    # Manages threads
    def manager(self, network=None):
        threads = []
        average_accuracy = np.array([])

        model = network[0]
        momentum_gradient = np.zeros(model.shape)

        queue = self.queue
        update_queue = self.update_queue
        
        for count in range(self.threads):
            receive_queue = Queue()

            thread = multiprocessing.Process(target=self.worker, args=(receive_queue, count,))
            threads.append([thread, receive_queue])

        for thread in threads:
            thread[0].start()


        count = 0
        old_accuracy = -1
        self.accuracy = None

        for _ in range(self.trials):
            
            count += 1
            self.generations += 1 
            self.model = model

            self.children = self.threads

            backup = open('model-training-data.py', 'r+').read()
            queue = self.queue

            with open('model-training-data.py', 'w+') as file:
                    try:
                        file.write(json.dumps([model.tolist(), network[1].tolist()] + network[2:]))
                    except Exception as e:
                        print(e)
                        file.write(backup)

            for thread in threads:
                thread[1].put(network)

            gradient = np.zeros(model.shape)

            for _ in range(self.threads):
                accuracy, gradient_map = queue.get()
                gradient += gradient_map

                if not self.accuracy:
                    self.accuracy = accuracy
                
                self.accuracy = (self.accuracy + accuracy) / 2
                self.children -= 1

            update_queue.put(self.accuracy)

            if self.accuracy < self.cost_limit:
                print(f"Cost Minimum Reached: {self.cost_limit}")
                os.system("PAUSE")

            model -= (gradient + (momentum_gradient * self.momentum)) * self.learning_rate / self.tests
            momentum_gradient = np.array(gradient)
            self.accuracy = None

    def worker(self, receieve_queue=None, thread_index=0):
        generations = 0
        start = 0


        while True:
            generations += 1
            model, heights, hidden_activation_function, output_activation_function, cost_function = receieve_queue.get()
            model = Model(
                model=model,
                heights=heights,
                hidden_function=hidden_activation_function, 
                output_function=output_activation_function, 
                cost_function=cost_function
            )

            gradient = np.zeros(model.model.shape)
            trials = self.tests
            points = 0
            games = 0

            load_amount = self.tests // self.threads

            for count in range(start + load_amount * thread_index, start + load_amount * (thread_index + 1)):
                
                count = count % (3 ** 9)

                game = Game()
                player = 0

                for _ in range(9):

                    inputs = game.grid.flatten()
                    oppenent = 1 - player

                    x_count = np.count_nonzero(inputs == 1)
                    y_count = np.count_nonzero(inputs == 2)

                    counts = [
                        x_count,
                        y_count
                    ]

                    if game.eval():
                        continue

                    best_moves = game.best_moves(player)[:1]

                    input_grid = np.zeros(18)
                    input_grid[:9][inputs == (player + 1)] = 1
                    input_grid[9:][inputs == (oppenent + 1)] = 1

                    expected_output = np.zeros(9)
                    expected_output[best_moves] = 1

                    model_outputs = model.eval(input_grid)

                    raw_prediction = np.array(model_outputs[-1])
                    open_slots = game.open_slots.tolist()

                    _gradient, average_cost = model.gradient(
                        model_outputs, 
                        expected_output
                    )

                    mask = np.ones(raw_prediction.shape, bool)
                    mask[open_slots] = False

                    prediction = np.array(raw_prediction)
                    prediction[mask] = -1

                    choice = np.argmax(prediction)

                    gradient += _gradient
                    points += 1 * average_cost
                    games += 1

                    random_int = np.random.rand()

                    if random_int < 0.25:
                        choice = random.choice(game.open_slots)

                    elif random_int < 0.75:
                        choice = random.choice(best_moves)

                    game.select(choice, player)
                    player = oppenent

            start += self.tests * self.threads

            if not trials:
                accuracy = 0
            else:
                accuracy = points / games

            self.queue.put([accuracy, gradient])
                    

    def build(self):
        inputs = 2
        shape = self.shape

        heights = np.full(self.length, self.height)
        max_height = max([inputs + 1, self.height + 1, max(shape.values()) if shape else -1])

        model = np.random.uniform(-2.5, 2.5, (self.length, max_height, max_height))

        if shape:
            heights[np.array([*shape.keys()])] = [*shape.values()]
            
        heights = np.append([18], heights)
        heights[-1] = 9

        return [model, heights, "tanh", "softmax", "cross_entropy"]

if __name__ == "__main__":
    Main(
        tests_amount = 220, # The length of the tests,
        generation_limit = 1000000, # The amount of generations the model will be trained through
        learning_rate = 0.02, # How fast the model learns, if too low the model will train very slow and if too high it won't train
        momentum_conservation = 0.001, # What percent of the previous changes that are added to each weight in our gradient descent
        weight_decay = 0.0,
        cost_limit = 0.0,
        dimensions = [
            [5, 36], 
            {
                
            }
        ],  # The length and height of the model
        threads = 10  # How many concurrent threads to be used
    )