class Data:
    """класс, который преобразует данные из тектового файла во вложенный список"""
    def __init__(self, name):
        with open(name) as f:
            self.d = f.readlines()
            for i in range(len(self.d)):
                self.d[i] = list(map(float, self.d[i].split()))

    def get_data(self):
        return self.d


"""data = Data('data_w_source.txt').get_data()
print(data)"""  # пример использования класса для получения подготовленных данных
