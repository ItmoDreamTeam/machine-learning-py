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

print(numpy.subtract(knn, svm))

p = wilcoxon(svm, knn)[1]
print("p = %f" % p)
