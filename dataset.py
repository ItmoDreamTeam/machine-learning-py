import csv
from matplotlib import pyplot


def visualize_chips_dataset(filename):
    x = [[], []]
    y = [[], []]
    with open(filename) as chips_dataset:
        reader = csv.DictReader(chips_dataset, fieldnames=('X', 'Y', 'Class'))
        for row in reader:
            if row["Class"] == '0':
                x[0].append(row["X"])
                y[0].append(row["Y"])
            else:
                x[1].append(row["X"])
                y[1].append(row["Y"])
    pyplot.plot(x[0], y[0], 'o')
    pyplot.plot(x[1], y[1], 'D')
    pyplot.show()


visualize_chips_dataset("chips1.csv")
