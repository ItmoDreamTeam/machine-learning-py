from scipy.stats import wilcoxon

knn = [
    0
]
svm = [
    1
]

T, p = wilcoxon(knn, svm)
print("T = %f" % T)
print("p = %f" % p)
