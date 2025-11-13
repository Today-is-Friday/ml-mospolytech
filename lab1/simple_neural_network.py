def neuralNetwork(inp, weight):
    prediction = inp * weight
    return prediction


out_1 = neuralNetwork(999, 0.10)
out_2 = neuralNetwork(420, 0.25)

print(out_1)
print(out_2)

## Измените входные данные и вес нейросети в коде. Запустите программу с новыми значениями и опишите,
## как это повлияло на выходные данные. Объясните, почему это произошло с точки зрения работы нейронной сети.

# У каждого входного сигнала есть свой параметр weight,
# т.к. не каждый параметр одинаково влияет на итоговый результат

inputs = [150, 160, 170, 180, 190]
for i in inputs:
    print("{}: {}".format(i, neuralNetwork(i, 0.35)))


## Как изменится выходная переменная? Почему?

# К каждому предикту будет добавлена константа bias.
# Он позвляет смещать нашу линейную функцию, если это требуется.


def neuralNetwork(inp, weight, bias):
    return inp * weight + bias


for i in inputs:
    print("{} (w bias): {}".format(i, neuralNetwork(i, 0.35, 20)))
