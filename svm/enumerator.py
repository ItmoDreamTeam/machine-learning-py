class Value:
    def __init__(self, min_value: int, max_value: int) -> None:
        self.min_value = min_value
        self.max_value = max_value
        self.value = min_value


class Enumerator:
    def __init__(self, values: [Value]) -> None:
        self.values = values

    def next(self) -> bool:
        increase_next = True
        for i in range(len(self.values) - 1, -1, -1):
            value = self.values[i]
            if increase_next:
                value.value += 1
                increase_next = False
            if value.value == value.max_value + 1:
                value.value = value.min_value
                increase_next = True
        return not increase_next
