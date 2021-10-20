# класс, который преобразует данные из тектового файла во вложенный список


class Data:
    def __init__(self, name):
        self.set_file(name)
        f = open(self.file)
        lines = f.read().split('\n')
        for i in range(len(lines)):
            lines[i] = list(map(float, lines[i].split()))
        self.d = lines.copy()

    def get_data(self):
        return self.d

    def set_file(self, name):
        self.file = f'{name}'


"""data = Data('data_w_source.txt').get_data()
print(data)"""  # пример использования класса для получения подготовленных данных
