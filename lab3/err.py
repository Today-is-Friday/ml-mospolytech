import numpy as np
from time import sleep


def neural_networks(inp, weights):
    return inp.dot(weights)


def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


true_prediction = 50
inp = [150, 40]
weights = [0.2, 0.3]
prediction = neural_networks(np.array(inp), weights)
print(get_error(true_prediction, prediction))


weights = [0.3, 0.2]
prediction = neural_networks(np.array(inp), weights)
print(get_error(true_prediction, prediction))


weights = [0.4, 0.1]
prediction = neural_networks(np.array(inp), weights)
print(get_error(true_prediction, prediction))


weights = [0.4, 0.2]
prediction = neural_networks(np.array(inp), weights)
print(get_error(true_prediction, prediction))


weights = [0.3, 0.19]
prediction = neural_networks(np.array(inp), weights)
print(get_error(true_prediction, prediction))


weights = [0.3, 0.15]
prediction = neural_networks(np.array(inp), weights)
print(get_error(true_prediction, prediction))


weights = [0.3, 0.13]
prediction = neural_networks(np.array(inp), weights)
print(get_error(true_prediction, prediction))

weights = [0.3, 0.125]
prediction = neural_networks(np.array(inp), weights)
print(get_error(true_prediction, prediction))
print(prediction)

weights = [15, 4]
prediction = neural_networks(np.array(inp), weights)


def sign_error(pred, pred_small, true):
    E = (pred - true) ** 2
    E_small = (pred_small - true) ** 2

    return -1 if E_small < E else 1


step = 0.001 * sign_error(
    neural_networks(np.array(inp), weights),
    neural_networks(np.array(inp), [weights[0] - 0.001, weights[1]]),
    true_prediction,
)

step_1 = 0.001 * sign_error(
    neural_networks(np.array(inp), weights),
    neural_networks(np.array(inp), [weights[0], weights[1] - 0.001]),
    true_prediction,
)

while get_error(true_prediction, prediction) > 0.001:
    weights[0] += step
    prediction = neural_networks(np.array(inp), weights)
    step *= sign_error(
        neural_networks(np.array(inp), weights),
        neural_networks(np.array(inp), [weights[0] - step, weights[1]]),
        true_prediction,
    )
    weights[1] += step_1
    prediction = neural_networks(np.array(inp), weights)
    step_1 *= sign_error(
        neural_networks(np.array(inp), weights),
        neural_networks(np.array(inp), [weights[0], weights[1] - step_1]),
        true_prediction,
    )

print(weights, prediction)
