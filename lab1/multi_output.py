def neuralNetwork(inp, weights):
    prediction = [0, 0]
    for i in range(len(weights)):
        prediction[i] = inp * weights[i]

    return prediction


print(neuralNetwork(10, [0.01, 0.04]))
print(neuralNetwork(10, [0.02, 0.05]))
print(neuralNetwork(10, [0.03, 0.05]))
print(neuralNetwork(10, [0.04, 0.05]))
print(neuralNetwork(10, [0.05, 0.05]))

inp = 6
x, y, w_x, w_y = 0, 0, 0.02, 0
while x < 0.5 or y < 0.5:
    if x < 0.5:
        w_x += 0.01

    if y < 0.5:
        w_y += 0.01

    x, y = neuralNetwork(inp, [w_x, w_y])
    print(x, y)
