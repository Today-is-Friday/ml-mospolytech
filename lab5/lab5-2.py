import numpy as np


def neural_networks(inp, weights):
    return inp.dot(weights)


def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


train_inp = np.array([[10, 5], [0, -5], [2, 6]])
weights = np.array([1.7, 0.1])
train_true_predictions = np.array([15, -5, 8])
learning_rate = 0.001

for i in range(367):
    error = 0
    delta = 0
    for j in range(len(train_inp)):
        current_inp = train_inp[j]
        true_prediction = train_true_predictions[j]
        prediction = neural_networks(current_inp, weights)
        error += get_error(true_prediction, prediction)
        delta += (prediction - true_prediction) * current_inp * learning_rate
        weights = weights - delta / len(train_inp)


test = neural_networks(np.array([12, 4]), weights), neural_networks(
    np.array([3, -8]), weights
)
true_result = 16, -5

for i in range(len(test)):
    if (test[i] - true_result[i]) < 0.01:
        print("Ответ верный")
    else:
        print("Ответ неверный")

print(test)
