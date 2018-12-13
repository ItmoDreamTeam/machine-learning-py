import numpy
from scipy.stats import wilcoxon

knn = [
    0.757895,
    0.754098,
    0.796460,
    0.756757,
    0.813559
]
svm = [
    0.857143,
    0.833333,
    0.785714,
    0.828829,
    0.831858
]
N = len(knn)

difference = list(numpy.subtract(svm, knn))

print(difference)
difference.sort(key=lambda v: abs(v))

rank = {}
for i, diff in enumerate(difference):
    rank[i] = diff

print([r for r in rank if rank[r] > 0])

Wn = abs(sum([r for r in rank if rank[r] < 0]))
Wp = abs(sum([r for r in rank if rank[r] > 0]))
W = min(Wp, Wn)

total = sum([diff for i, diff in enumerate(difference) if rank[i] > 0])
mu = N * (N + 1) / 4
sigma = (N * (N + 1) * (2 * N + 1) / 24) ** 0.5
z = (W - mu) / sigma
print("z", z)

p = wilcoxon(svm, knn)[1]
print("p = %f" % p)
