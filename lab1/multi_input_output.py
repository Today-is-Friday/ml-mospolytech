def neuralNetwork(inp, weights):
    # weights - является вектором векторов, т.е. мы получаем двумерный массив
    # значений весовых коэффициентов или матрицу
    prediction = [0] * len(weights)  # Упростим запись количества прогнозов
    for i in range(len(weights)):
        ws = 0  # средневзвешенное значение i выходного нейрона

        for j in range(len(inp)):
            ws += inp[j] * weights[i][j]

        prediction[i] = ws

    return prediction


inp = [50, 165, 45]
weights_1 = [0.2, 0.1, 0.65]
weights_2 = [0.3, 0.1, 0.7]
weights_3 = [0.5, 0.4, 0.34]
weights_4 = [0.4, 0.2, 0.1]
weights = [weights_1, weights_2, weights_3, weights_4]
# print(weights) #Получим след весовые коэффициенты выходных нейронов [[0.2, 0.1], [0.3, 0.1]]

print(neuralNetwork(inp, weights))


# Добавилось еще одно значение среднезвешенного. Остальные значения ни как не изменились
print()
weights_1[0] += 0.1
print(neuralNetwork(inp, weights))
weights_1[2] += 0.05
print(neuralNetwork(inp, weights))


weights_1 = [0.2, 0.1, 0.65]
weights_2 = [0.3, 0.1, 0.7]

x, y, z, t = neuralNetwork(inp, weights)

while x != y:
    if weights_1[0] < weights_2[0]:
        weights_1[0] += 0.01

    if weights_1[1] < weights_2[1]:
        weights_1[1] += 0.01

    if weights_1[1] < weights_2[1]:
        weights_1[1] += 0.01

    if weights_1[0] > weights_2[0]:
        weights_2[0] += 0.01

    if weights_1[1] > weights_2[1]:
        weights_2[1] += 0.01

    if weights_1[1] > weights_2[1]:
        weights_2[1] += 0.01

    x, y, z, t = neuralNetwork(inp, weights)

print(x, y, z, t)
