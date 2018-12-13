import numpy as np
from matplotlib import pyplot as plt
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers

from classification.kNN import Point
from classification.kNN import Metrics
from classification.kNN import read_dataset


def plot_points(points, title):
    plt.title(title)
    for point in points:
        plt.plot(point.x, point.y, "xr" if point.category else ".b")
    plt.show()


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


def sum_metrics(metrics: [Metrics]) -> Metrics:
    metric = Metrics()
    for m in metrics:
        metric += m
    return metric


# Kernel
def kernel(x1, x2):
    def poly_kernel(x1, x2):
        return (1 + np.dot(x1, x2)) ** 2

    def rbf_kernel(x1, x2):
        distance = np.sqrt(np.sum((x1 - x2) ** 2))
        return np.exp(-(distance ** 2 / 2))

    # return poly_kernel(x1, x2)
    return rbf_kernel(x1, x2)


def train(X, y, C=10):
    n_samples, n_features = X.shape
    H = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            H[i, j] = kernel(X[i], X[j])

    P = cvxopt_matrix(np.outer(y, y) * H, tc='d')
    q = cvxopt_matrix(np.ones(n_samples) * -1)
    A = cvxopt_matrix(y, (1, n_samples), tc='d')
    b = cvxopt_matrix(0.0, tc='d')
    G_max = np.identity(n_samples) * -1
    G_min = np.identity(n_samples)
    G = cvxopt_matrix(np.vstack((G_max, G_min)))
    h_max = cvxopt_matrix(np.zeros(n_samples))
    h_min = cvxopt_matrix(np.ones(n_samples) * C)
    h = cvxopt_matrix(np.vstack((h_max, h_min)))

    solution = cvxopt_solvers.qp(P, q, G, h, A, b)
    lm = np.ravel(solution['x'])
    idx = lm > 1e-5
    ind = np.arange(len(lm))[idx]
    lagr_multipliers = lm[idx]
    sv = X[idx]
    sv_labels = y[idx]
    bias = 0
    for n in range(len(lagr_multipliers)):
        bias += sv_labels[n]
        bias -= np.sum(lagr_multipliers * sv_labels * H[ind[n], idx])
    bias /= len(lagr_multipliers)
    return lagr_multipliers, sv, sv_labels, bias


def predict(X, lagr_multipliers, sv, sv_labels, bias) -> bool:
    y_pred = np.zeros(len(X))
    for i in range(len(X)):
        prediction = 0
        for lagr_multipliers, sv_labels, sv in zip(lagr_multipliers, sv_labels, sv):
            prediction += lagr_multipliers * sv_labels * kernel(X[i], sv)
        y_pred[i] = prediction
    return np.sign(y_pred + bias) > 0


def categorize_batch(train_batch: [Point], test_points: [Point]) -> Metrics:
    # Split to X and y
    X = np.array([[point.x, point.y] for point in train_batch])
    y = np.array([1 if point.category else -1 for point in train_batch])

    # Train
    lagr_multipliers, sv, sv_labels, bias = train(X, y)

    # Calculate metrics
    metrics = Metrics()
    for test_point in test_points:
        computed_category = predict([[test_point.x, test_point.y]], lagr_multipliers, sv, sv_labels, bias)
        metrics.add(test_point.category, computed_category)
    return metrics


def categorize_all(points: [Point], batch_count: int) -> [Metrics]:
    metrics = []
    batch_size = int(len(points) / batch_count)
    for i in range(batch_count):
        start = i * batch_size
        end = (i + 1) * batch_size
        test = points[start:end]
        train = points[:start] + points[end:]
        metrics.append(categorize_batch(train, test))
    return metrics


# Read dataset
points = read_dataset("../classification/chips1.csv", shuffle=True)
# plot_points(points, "Original data")

# Solve
metrics_list = categorize_all(points, batch_count=10)
print_metrics(sum_metrics(metrics_list), "Metrics:")
