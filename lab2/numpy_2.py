import numpy as np


def neuralNetwork(inp, weights):
    prediction_h = inp.dot(weights[0])
    prediction_mid = prediction_h.dot(weights[1])
    prediction_out = prediction_mid.dot(weights[2])

    return prediction_out


inp = np.array([23, 45])
# weight_h_1 = [0.4, 0.1]
# weight_h_2 = [0.3, 0.2]
# weights_h_3 = [0.6, 0.2]
# weight_out_1 = [0.4, 0.1]
# weight_out_2 = [0.3, 0.1]
# weights_out_3 = [0.7, 0.4]
# weights_h = np.array([weight_h_1, weight_h_2, weights_h_3]).T
# weights_mid = np.array([[1] * 3] * 3)
# weights_out = np.array([weight_out_1, weight_out_2, weights_out_3])

weights_h = np.random.randint(0, 10, size=(2, 3)) / 10
weights_mid = np.random.randint(0, 10, size=(3, 3)) / 10
weights_out = np.random.randint(0, 10, size=(3, 2)) / 10

weights = [weights_h, weights_mid, weights_out]


print(neuralNetwork(inp, weights))
