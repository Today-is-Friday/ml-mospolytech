def neuralNetwork(inp, weights):
    prediction_h = [0] * len(weights[0])
    for i in range(len(weights[0])):
        ws = 0
        for j in range(len(inp)):
            ws += inp[j] * weights[0][i][j]

        prediction_h[i] = ws

    prediction_out = [0] * len(weights[1])  # Прогноз для выходного слоя
    for i in range(len(weights[1])):
        ws = 0  # средневзвешенное значение i выходного нейрона
        for j in range(len(prediction_h)):
            ws += prediction_h[j] * weights[1][i][j]

        prediction_out[i] = ws

    return prediction_out


inp = [23, 45]
weight_h_1 = [0.4, 0.1]  # весовые коэффициенты для первого нейрона скрытого слоя
weight_h_2 = [0.3, 0.2]  # весовые коэффициенты для второго нейрона скрытого слоя
weight_out_1 = [
    0.4,
    0.1,
]  # весовые коэффициенты для связи нейронов скрытого слоя и выходного
weight_out_2 = [
    0.3,
    0.1,
]  # весовые коэффициенты для связи нейронов скрытого слоя и выходного
weights_h = [weight_h_1, weight_h_2]
weights_out = [weight_out_1, weight_out_2]
weights = [weights_h, weights_out]
print(neuralNetwork(inp, weights))
