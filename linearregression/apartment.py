from csv_reader import CsvReader


class ApartmentCsvReader(CsvReader):

    @staticmethod
    def read_apartments() -> ["Apartment"]:
        reader = ApartmentCsvReader()
        reader.read("prices.csv")
        return reader.apartments

    def __init__(self):
        self.apartments = []

    def next_element(self, e: {str, object}):
        self.apartments.append(Apartment(e["area"], e["rooms"], e["price"]))


class Apartment:

    def __init__(self, area: float, rooms: int, price: float):
        self.__area = float(area)
        self.__rooms = int(rooms)
        self.__price = float(price)

    def __str__(self) -> str:
        return "{area=%.1f,rooms=%i,price=%.1f}" % (self.__area, self.__rooms, self.__price)

    def __repr__(self):
        return str(self)
