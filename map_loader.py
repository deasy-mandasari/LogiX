import numpy as np
import csv

def load_csv_map(path):
    grid = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            grid.append([int(x) for x in row])
    return np.array(grid)