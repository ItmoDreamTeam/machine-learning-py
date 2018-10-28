import csv


class CsvReader:

    def read(self, filename: str) -> None:
        with open(filename) as file:
            reader = csv.DictReader(file)
            for e in reader:
                self.next_element(e)

    def next_element(self, e: {str, object}) -> None:
        pass
