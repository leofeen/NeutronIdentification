from prepare_data import Data  # Данные берутся по уже готовому формату


def find_max(string):
    # Поиск максимума, он же пик
    max_coord = [0, 0]  # координаты - индекс и величина

    # Процесс поиска максимума
    for i in range(len(string)):
        if max_coord[1] < string[i]:
            max_coord = [i, string[i]]

    return max_coord


def calculating_slope_left(string, max_coord, slope_left):
    # Левая секущая
    tau_left = 10  # коэффициент пропорциональности для левой секущей
    slope_left_coordinat = [0, 0, 0]

    while slope_left_coordinat[2] == 0:
        # Некоторая секущая, которая круче slope_left
        new_slope = tau_left*slope_left

        # Поиск секущей, которая имеет такой же или больший угол наклона
        for i in range(1, max_coord[0]):
            next_dot = string[i]
            previous_dot = string[i-1]

            # Коэффициент угла наклона секущей
            derivative = next_dot - previous_dot
            if derivative >= new_slope:  # угол больше
                slope_left_coordinat = [i,  next_dot, derivative]
                break

        # если ничего не подошло, то ищем менее крутую секущую
        tau_left -= 0.5
    return slope_left_coordinat


def calculating_slope_right(string, max_coord, slope_right):
    # Правая секущая

    tau_right = 100  # коэффициент пропорциональности для правой секущей
    slope_right_coordinates = [0, 0, 0]

    while slope_right_coordinates[2] == 0:
        # Некоторая секущая, которая круче slope_right
        new_slope = tau_right * slope_right

        # Поиск секущей, которая имеет такой же или меньший угол наклона
        for i in range(1900, max_coord[0], -1):
            next_dot = string[i]
            previous_dot = string[i - 1]

            # Коэффициент угла наклона секущей
            derivative = next_dot - previous_dot
            if derivative <= new_slope:  # угол меньше
                slope_right_coordinates = [i, previous_dot, derivative]
                break

        # если ничего не подошло, то ищем менее крутую секущую
        tau_right -= 0.5

    return slope_right_coordinates


def surface_near_max(string, max_dot, left_dot, right_dot):
    # Данная функция считает площадь под кривой в области около пика, а также площадь самого пика.

    # Площадь под всей кривой
    integral_surface = 0

    # Площадь пика
    figure_surface = 0

    # Длина отрезка между правой и левой секущими
    length = (right_dot - left_dot)

    # Левая граница исследуемой кривой
    left_side = max_dot - length
    # Проверка того, что не вышли за границы массива, иначе первое значение
    if left_side < 0:
        left_side = 0

    # Правая граница исследуемой кривой
    right_side = max_dot + 2*length

    # Проверка того, что не вышли за границы массива, иначе последнее значение
    if right_side > len(string):
        right_side = len(string)

    # Процесс вычисления искомых площадей
    for i in range(left_side, right_side):
        integral_surface += string[i]

        # С этого момента появляется пик
        if left_dot < i < right_dot:
            figure_surface += string[i]

    return integral_surface, figure_surface


def answering(whole_surface, pike_surface, answer):
    our_answer = [0.0, False]
    # Ответ определяется на основе того предположения,
    # что отношение площади пика ко все площади меньше некоторой константы, если нейтрон есть
    if pike_surface/whole_surface < 78/125:
        our_answer[0] = 1.0

    # Сверяемся с настоящим ответом
    if our_answer[0] == answer:
        our_answer[1] = True

    return our_answer


def main(string, answer, flag):
    """ Интегральная модель """

    # Поиск точки максимума
    maximum_coord = find_max(string)
    """ 
    Идея локализации пика.
    Найти две точки, секущие которых имеют больший угол в направлении на пика 
    А соответственно в этих точках начинается/заканчивается пик.
    """

    """ Поиск секущих. Начало. """
    # Левая секущая
    slope_left = (maximum_coord[1] - string[0])/maximum_coord[0]
    # Первая секущая слева которая круче чем данная

    # Здесь понятее "круче", означает, что модуль тангенся угла наклона секущей больше другого.
    slope_left_coord = calculating_slope_left(string, maximum_coord, slope_left)

    # Правая секущая
    slope_right = (string[1900] - maximum_coord[1]) / (1900 - maximum_coord[0] + 1)  # 1900 - т.к. это половина списка
    # Первая секущая с права которая круче чем данная
    slope_right_coord = calculating_slope_right(string, maximum_coord, slope_right)
    """ Поиск секущих. Конец. """

    # Площадь под кривой integral_surface - ОКОЛО пика, figure_surface - площадь самого пика
    integral_surface, figure_surface = surface_near_max(string, maximum_coord[0], slope_left_coord[0], slope_right_coord[0])

    # Ответ
    pred_answer = answering(integral_surface, figure_surface, answer)

    # Упаковка собранных данных
    solved_data = [integral_surface, figure_surface, slope_left_coord, maximum_coord, slope_right_coord, pred_answer]

    # Какой "режим" был включен 0- только ответ, 1 - ответ + доп.данные
    if flag:
        # Ответ + доп. данные
        if pred_answer[1]:
            # Если ответ верный, сходится с табличными данными
            return 1, solved_data
        else:
            # Если ответ не верный, не сходится с табличными данными
            return 0, solved_data
    else:
        # Только Ответ
        return pred_answer[0]


def training_accuracy(data, answers):
    Truth = 0
    length = len(data)
    ans_0, ans_1 = 0, 0
    frac_0, frac_1 = 0, 0

    for index_string in range(length):
        data_string, data_answer = data[index_string], answers[index_string]
        check, solved_data = main(data_string, data_answer, 1)
        if check:
            Truth += 1
        else:
            print("\nId =", index_string)
            print("Ответ модели - {0}, на самом деле - {1}".format(solved_data[5], data_answer))
            print("Левая грань пика - {0}, Пик - {1}, Правая грань пика - {2}".format(solved_data[2], solved_data[3], solved_data[4]))
            frac = solved_data[1] / solved_data[0]
            print("Полная площадь = {0},\tПлощадь пика = {1}, Отношение = {2}\n".format(solved_data[0], solved_data[1], frac))
            if data_answer == 0.0:
                ans_0 += 1
                frac_0 += frac
            else:
                ans_1 += 1
                frac_1 += frac
    accuracy = 100 * (Truth / length)
    print("\nТочность", accuracy, "%")
    print("Должен быть '0'- {0} шт, S_whole/S_fig = {1}\t Must be '1'- {2}, S_whole/S_fig = {3}\t"
          .format(ans_0, frac_0 / ans_0, ans_1, frac_1 / ans_1))
    return accuracy


def output(data_string, answer_string):
    answer_for_one_string = main(data_string, answer_string, 0)
    print(int(answer_for_one_string))


if __name__ == '__main__':

    input_data = Data('data_w_source.txt', "class_labels.txt")
    data = input_data.get_data()
    answers = input_data.get_answers()

    """ Для одной строки, конкретной строки"""
    # INDEX = 0
    # output(data[INDEX], answers[INDEX])

    """ Для всех строк"""
    # for INDEX in range(len(data)):
    #     output(data[INDEX], answers[INDEX])

    """ Все строки сразу + доп.данные """
    training_accuracy(data, answers)

    """ поиск "идеального" параметра для отношения между площадями """
    # global parametr
    # parametr = 0.4
    # ideal_truth = 0
    # d = 1/8192
    # while parametr < 0.8:
    #     pred_truth = output(data, answers)
    #     if ideal_truth < pred_truth:
    #             ideal_truth = pred_truth
    #             ideal_parametr = parametr
    #     print(pred_truth, parametr)
    #     parametr += d
    # print("Лучшая точность = {0}, параметр = {1}".format(ideal_truth, ideal_parametr))
