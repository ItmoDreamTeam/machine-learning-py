import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

import classification.kNN
from classification.kNN import Point
from classification.kNN import Metrics
from classification.kNN import read_dataset


def print_results(w, b, title):
    print("\n %s" % title)
    print("w = %s" % w.flatten())
    print("b = %f" % b[0])


def print_metrics(metrics, title):
    print("\n %s" % title)
    print("-------------------------------------------------------------")
    print("|                 | Predicted positive | Predicted negative |")
    print("|-----------------|--------------------|--------------------|")
    print("| Actual positive | %18d | %18d |" % (metrics.true_positive(), metrics.false_negative()))
    print("| Actual negative | %18d | %18d |" % (metrics.false_positive(), metrics.true_negative()))
    print("-------------------------------------------------------------")
    print("Accuracy = %f" % metrics.accuracy())
    print("Precision = %f" % metrics.precision())
    print("Recall = %f" % metrics.recall())
    print("F measure = %f" % metrics.f_measure())





points = read_dataset("../classification/chips1.csv", shuffle=False)
# for point in points:
#     point.x **= 2
#     point.y **= 2

NUMBER_OF_BATCHES = 3
svm_metrics = categorize_all(points, NUMBER_OF_BATCHES)
knn_metrics = classification.kNN.categorize_all(points, NUMBER_OF_BATCHES, 3)

print_metrics(average_metrics(svm_metrics), "SVM metrics")
print_metrics(average_metrics(knn_metrics), "kNN metrics")

# Wilcoxon
from scipy.stats import wilcoxon

_, p = wilcoxon([m.f_measure() for m in knn_metrics], [m.f_measure() for m in svm_metrics])
print("p", p)
