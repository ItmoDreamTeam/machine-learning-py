import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

from classification.kNN import Metrics
from classification.kNN import read_dataset

X_LEFT = -0.5
X_RIGHT = 2
Y_UP = 2
Y_DOWN = -0.5


def show_points(points):
    for point in points:
        if point.category == 0:
            plt.plot(point.x, point.y, "go")
        else:
            plt.plot(point.x, point.y, "co")
    plt.show()


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def poly_kernel(x1, x2):
    return (1 + np.dot(x1, x2)) ** 2


def rbf_kernel(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return np.exp(-(distance ** 2 / 2))


# Read dataset
points = read_dataset("../classification/chips1.csv")
show_points(points)

# Apply some transform
for point in points:
    point.x **= 2
    point.y **= 2
show_points(points)

# Data set
x_neg = np.array([[point.x, point.y] for point in points if not point.category])
y_neg = np.array([-1 for i in x_neg])
x_pos = np.array([[point.x, point.y] for point in points if point.category])
y_pos = np.array([1 for i in x_pos])
x1 = np.linspace(X_LEFT, X_RIGHT)
x = np.vstack((np.linspace(X_LEFT, X_RIGHT), np.linspace(X_LEFT, X_RIGHT)))
X = np.vstack((x_pos, x_neg))
y = np.concatenate((y_pos, y_neg))

fig = plt.figure(figsize=(10, 10))
plt.scatter(x_neg[:, 0], x_neg[:, 1], marker='x', color='r', label='Negative -1')
plt.scatter(x_pos[:, 0], x_pos[:, 1], marker='o', color='b', label='Positive +1')
plt.xlim(X_LEFT, X_RIGHT)
plt.ylim(Y_DOWN, Y_UP)
plt.xticks(np.arange(X_LEFT, X_RIGHT, step=1))
plt.yticks(np.arange(Y_DOWN, Y_UP, step=1))

# Lines
plt.axvline(0, color='black', alpha=.5)
plt.axhline(0, color='black', alpha=.5)

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

plt.legend(loc='lower right')
# plt.show()


### SOLUTION ###
# Initializing values and computing H. Note the 1. to force to float type
C = 10
m, n = X.shape
y = y.reshape(-1, 1) * 1.
X_dash = y * X


# H = np.dot(X_dash, X_dash.T) * 1.

def make_kernel_tricks(kernel):
    n_samples = len(points)
    kernel_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            kernel_matrix[i, j] = kernel(X[i], X[j])
    return kernel_matrix


H = make_kernel_tricks(linear_kernel)

# Converting into cvxopt format - as previously
P = cvxopt_matrix(np.outer(y, y) * H, tc='d')
q = cvxopt_matrix(-np.ones((m, 1)))
G = cvxopt_matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
A = cvxopt_matrix(y.reshape(1, -1))
b = cvxopt_matrix(np.zeros(1))

# Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])

# ==================Computing and printing parameters===============================#
w = ((y * alphas).T @ X).reshape(-1, 1)
S = (alphas > 1e-4).flatten()
b = y[S] - np.dot(X[S], w)

# Display results
print('Alphas = ', alphas[alphas > 1e-4])
print('w = ', w.flatten())
print('b = ', b[0])

# Display results
plt.plot(x1, - w[0] / w[1] * x1 - b[0] / w[1], color='darkblue')
plt.show()

clf = SVC(C=10, kernel='linear')
clf.fit(X, y.ravel())

print("*********")
print('w = ', clf.coef_)
print('b = ', clf.intercept_)
print('Indices of support vectors = ', clf.support_)
print('Support vectors = ', clf.support_vectors_)
print('Number of support vectors for each class = ', clf.n_support_)
print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))

etalon = clf.predict(X)

metrics = Metrics()
etalon_metrics = Metrics()
for i, p in enumerate(points):
    predicted = p.x * w[0] + p.y * w[1] + b[0] > 0
    etalon_predicted = etalon[i] > 0
    metrics.add(p.category, predicted)
    etalon_metrics.add(p.category, etalon_predicted)

print("Accuracy = %f" % metrics.accuracy())
print("Precision = %f" % metrics.precision())
print("Recall = %f" % metrics.recall())

print("Accuracy = %f" % etalon_metrics.accuracy())
print("Precision = %f" % etalon_metrics.precision())
print("Recall = %f" % etalon_metrics.recall())
