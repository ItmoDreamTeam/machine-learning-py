import csv
import random


class Point:
    def __init__(self, x: float, y: float, category: bool):
        self.x = float(x)
        self.y = float(y)
        self.category = bool(category)

    def find_closest(self, points: [], count: int):
        distances = map(lambda point: {"point": point, "distance": self.distance(point)}, points)
        distances = sorted(distances, key=lambda distance: distance["distance"])
        return list(map(lambda distance: distance["point"], distances[:count]))

    def distance(self, point):
        return ((self.x - point.x) ** 2 + (self.y - point.y) ** 2) ** 0.5

    def __str__(self):
        return "{x:%f,y:%f,c:%s}" % (self.x, self.y, self.category)

    def __repr__(self):
        return str(self)


class Metrics:
    def __init__(self):
        self.true_values = []
        self.computed_values = []

    def add(self, true_value: bool, computed_value: bool):
        self.true_values.append(true_value)
        self.computed_values.append(computed_value)

    def margins(self) -> [int]:
        return [0 if self.true_values[i] == self.computed_values[i] else 1 for i in range(len(self.true_values))]

    def accuracy(self) -> float:
        return sum(self.true_values[i] == self.computed_values[i] for i in range(len(self.true_values))) \
               / len(self.true_values)

    def precision(self) -> float:
        if self.true_positive() + self.false_positive() == 0:
            return -1
        return self.true_positive() / (self.true_positive() + self.false_positive())

    def recall(self):
        if self.true_positive() + self.false_negative() == 0:
            return -1
        return self.true_positive() / (self.true_positive() + self.false_negative())

    def f_measure(self):
        if self.precision() + self.recall() == 0:
            return -1
        return 2 * self.precision() * self.recall() / (self.precision() + self.recall())

    def true_positive(self) -> int:
        result = 0
        for i in range(len(self.true_values)):
            if self.true_values[i] and self.computed_values[i]:
                result += 1
        return result

    def true_negative(self) -> int:
        result = 0
        for i in range(len(self.true_values)):
            if not self.true_values[i] and not self.computed_values[i]:
                result += 1
        return result

    def false_positive(self) -> int:
        result = 0
        for i in range(len(self.true_values)):
            if not self.true_values[i] and self.computed_values[i]:
                result += 1
        return result

    def false_negative(self) -> int:
        result = 0
        for i in range(len(self.true_values)):
            if self.true_values[i] and not self.computed_values[i]:
                result += 1
        return result

    def __add__(self, other):
        result = Metrics()
        result.true_values = self.true_values + other.true_values
        result.computed_values = self.computed_values + other.computed_values
        return result

    def __str__(self):
        return "Accuracy: %f\nPrecision: %f\nRecall: %f\nF-measure: %f" % \
               (self.accuracy(), self.precision(), self.recall(), self.f_measure())

    def __repr__(self):
        return str(self)


def read_dataset(filename: str, shuffle: bool = False) -> [Point]:
    points = []
    with open(filename) as dataset_file:
        reader = csv.DictReader(dataset_file, fieldnames=('X', 'Y', 'Class'))
        for point in reader:
            points.append(Point(point["X"], point["Y"], point["Class"] == "1"))
    if shuffle:
        random.shuffle(points)
    return points


def categorize_point(train_batch: [Point], test_point: Point, k: int) -> bool:
    closest_points = test_point.find_closest(train_batch, k)
    categories = list(map(lambda point: point.category, closest_points))
    categories_count = {}
    for category in categories:
        if category in categories_count:
            categories_count[category] += 1
        else:
            categories_count[category] = 1
    return sorted(categories_count, key=lambda key: categories_count[key], reverse=True)[0]


def categorize_batch(train_batch: [Point], test_points: [Point], k: int) -> Metrics:
    metrics = Metrics()
    for test_point in test_points:
        computed_category = categorize_point(train_batch, test_point, k)
        metrics.add(test_point.category, computed_category)
    return metrics


def categorize_all(points: [Point], batch_count: int, k: int) -> [Metrics]:
    metrics = []
    batch_size = int(len(points) / batch_count)
    for i in range(batch_count):
        start = i * batch_size
        end = (i + 1) * batch_size
        test = points[start:end]
        train = points[:start] + points[end:]
        metrics.append(categorize_batch(train, test, k))
    return metrics


def train(points: [Point]) -> Metrics:
    max_f_measure = 0
    best_metrics = None
    for k in range(1, len(points)):
        for batch_count in range(2, len(points) - 1):
            metrics = categorize_all(points, batch_count, k)
            f_measure = metrics.f_measure()
            if f_measure > max_f_measure or k % 10 == 0 and batch_count == len(points) - 2:
                print("%.1f%%" % (100 * (k + 1) / len(points)))
            if f_measure > max_f_measure:
                max_f_measure = f_measure
                best_metrics = metrics
                print("k=%d, batch_count=%d, F=%f\n%s\n" % (k, batch_count, f_measure, metrics))
    return best_metrics


if __name__ == '__main__':
    points = read_dataset("chips1.csv")
    train(points)
