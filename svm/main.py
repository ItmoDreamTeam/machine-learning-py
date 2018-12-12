import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

from classification.kNN import Metrics
from classification.kNN import read_dataset

# Variables' ranges
X_FROM, X_TO = -0.1, 1.3
Y_FROM, Y_TO = -0.1, 1.3


def plot_points(points, title):
    plt.title(title)
    for point in points:
        plt.plot(point.x, point.y, "xr" if point.category else ".b")
    plt.show()


def print_results(w, b, title):
    print("\n %s" % title)
    print("w = %s" % w.flatten())
    print("b = %f" % b[0])


def print_metrics(metrics, title):
    print("\n %s" % title)
    print("Accuracy = %f" % metrics.accuracy())
    print("Precision = %f" % metrics.precision())
    print("Recall = %f" % metrics.recall())
    print("F measure = %f" % metrics.f_measure())


# Read dataset
points = read_dataset("../classification/chips1.csv")
plot_points(points, "Original data")

# Transform
for point in points:
    point.x **= 2
    point.y **= 2
plot_points(points, "Squared data")

# Split to X and y
X = np.array([[point.x, point.y] for point in points])
y = np.array([1 if point.category else -1 for point in points])

# Prepare solver
C = 10
elements_number = X.shape[0]
y = y.reshape(-1, 1) * 1.
X_dash = y * X
H = np.dot(X_dash, X_dash.T) * 1.
P = cvxopt_matrix(H)
q = cvxopt_matrix(-np.ones((elements_number, 1)))
G = cvxopt_matrix(np.vstack((np.eye(elements_number) * -1, np.eye(elements_number))))
h = cvxopt_matrix(np.hstack((np.zeros(elements_number), np.ones(elements_number) * C)))
A = cvxopt_matrix(y.reshape(1, -1))
b = cvxopt_matrix(np.zeros(1))

# Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])
w = ((y * alphas).T @ X).reshape(-1, 1)
S = (alphas > 1e-4).flatten()
b = y[S] - np.dot(X[S], w)
print_results(w, b, "Our results")

# Compare with SkLearn
svc = SVC(C=C, kernel='linear')
svc.fit(X, y.ravel())
print_results(svc.coef_, svc.intercept_, "SkLearn results")

# Calculate metrics
our_metrics = Metrics()
sklearn_metrics = Metrics()
sklearn_prediction = [e > 0 for e in svc.predict(X)]
for index, point in enumerate(points):
    our_prediction = point.x * w[0] + point.y * w[1] + b[0] > 0
    our_metrics.add(point.category, our_prediction)
    sklearn_metrics.add(point.category, sklearn_prediction[index])
print_metrics(our_metrics, "Our metrics")
print_metrics(sklearn_metrics, "SkLearn metrics")

# Visualize results
plt.figure(figsize=(10, 10))

# Points
for point in points:
    plt.plot(point.x, point.y, "xr" if point.category else ".b")

# Separating lines
xs = np.linspace(X_FROM, X_TO)
plt.plot(xs, - w[0] / w[1] * xs - b[0] / w[1], color='green')
plt.plot(xs, - svc.coef_[0][0] / svc.coef_[0][1] * xs - svc.intercept_[0] / svc.coef_[0][1], color='magenta')

plt.xlim(X_FROM, X_TO)
plt.ylim(Y_FROM, Y_TO)
plt.axvline(0, color='black')
plt.axhline(0, color='black')
plt.show()
