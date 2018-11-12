import os

from sklearn.naive_bayes import GaussianNB


def read_data_set(directory: str) -> [[int], [bool]]:
    data_set = []
    for filename in os.listdir(directory):
        with open(directory + "/" + filename) as file:
            words = []
            for line in file.readlines():
                for word in line.split():
                    if word.isdigit():
                        words.append(int(word))
            is_spam = "spm" in filename
            data_set.append([words, [is_spam]])
    return data_set


def transfer_words_into_quantities(raw_data_set: [[int], [bool]]) -> [[int], [bool]]:
    max = 30000
    data_set = []
    for item in raw_data_set:
        words = item[0]
        words_quantities = [0 for i in range(max)]
        for word in words:
            words_quantities[word] += 1
        data_set.append([words_quantities, item[1]])
    return data_set


def extract_xy(data_set: [[int], [bool]]) -> ([[int]], [bool]):
    x = []
    y = []
    for item in data_set:
        x.append(item[0])
        y.append(item[1][0])
    return x, y


def train(train: [[int], [bool]], test: [[int], [bool]]) -> float:
    print("train: %d, test: %d" % (len(train), len(test)))
    nb = GaussianNB()
    x_train, y_train = extract_xy(train)
    nb.fit(x_train, y_train)
    x_test, y_test = extract_xy(test)
    return sum(nb.predict(x_test) == y_test) / len(y_test)


def cross_validate(sets: [[[int], [bool]]]) -> [float]:
    success = []
    for set in sets:
        tr = []
        for train_set in sets:
            if not set == train_set:
                for train_set_item in train_set:
                    tr.append(train_set_item)
        success.append(train(set, tr))
    return success


ROOT_DIR = "./messages/"
sets = []
for part_name in os.listdir(ROOT_DIR):
    raw_data_set = read_data_set(ROOT_DIR + part_name)
    data_set = transfer_words_into_quantities(raw_data_set)
    sets.append(data_set)
success = cross_validate(sets)
print("probabilities: %s\nAverage = %.1f%%" % (success, sum(success) / len(success) * 100))
