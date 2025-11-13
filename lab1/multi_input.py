def neuralNetwork(inps, weights):
    prediction = 0
    calculations = []
    for i in range(len(weights)):
        prediction += inps[i] * weights[i]
        calculations.append(inps[i] * weights[i])

    return prediction, calculations


out_1 = neuralNetwork([150, 40], [0.3, 0.4])
out_2 = neuralNetwork([80, 60], [0.2, 0.4])

print(out_1)
print(out_2)
