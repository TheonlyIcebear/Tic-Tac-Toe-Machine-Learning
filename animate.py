import matplotlib.pyplot as plt, json

plt.ion()
    
while True:
    try:
        best_overtime = json.load(open('generations.json', 'r+'))
    except:
        continue
    
    plt.plot([i + 1 for i in range(len(best_overtime))], best_overtime, )
    plt.draw()
    plt.pause(0.0001)
    plt.clf()