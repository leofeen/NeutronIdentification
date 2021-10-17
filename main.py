import numpy as np
import matplotlib.pyplot as mp


def main():
    dt = 8*10**(-3)
    sum = []
    with open("data_w_source.txt", 'r') as amplituda:
        i = 0
        for line in amplituda:
            i += 1
            A = list(map(float, amplituda.readline().split('\t')))
            pre_sum = 0
            for x in A:
                pre_sum += x*dt
            sum.append(round(pre_sum, 2))
    print(sum)
    for i in range(len(sum)):
        if sum[i] > 5:  # выбрано на основе значения для которого
            sum[i] = [sum[i], 1]
        else:
            sum[i] = [sum[i], 0]
    with open("class_labels.txt", 'r') as is_neutron:
        i = 0
        for line in is_neutron:
            try:
                if int(line) == sum[i][1]:
                    sum[i].append(True)
                else:
                    sum[i].append(False)
                i += 1
            except IndexError:
                print("Out of range")
                break

        print(sum)
        print("Далее выводятся элементы которые не свопадают со значениями в файле class_labels \n"
              "Формат вывода- номер линии, [значение амплитуды*10^-6, найден нейтрон программой? "
              "(0 - нет, 1- да), false - не совпало с итоговым файлом]")
        n = 0
        for i in range(len(sum)):
            if sum[i][2]==False:
                print(i, sum[i])
                n+=1
        print(n/len(sum)*100)

if __name__ == "__main__":
    main()
