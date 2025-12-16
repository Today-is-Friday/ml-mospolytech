import numpy as np


def neural_networks(inp, weights):
    return inp.dot(weights)


def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


inp = np.array([150, 40])
weights = np.array([0.2, 0.3])
true_prediction = 1
learning_rate = 0.00001  # подбираем необходимую скорость обучения

for i in range(30):
    prediction = neural_networks(inp, weights)
    error = get_error(true_prediction, prediction)
    print("Prediction: %.10f, Weights: %s, Error: %.20f" % (prediction, weights, error))
    delta = (prediction - true_prediction) * inp * learning_rate
    weights = weights - delta


inp = np.array([150, 40])
weights = np.array([0.2, 0.3])
true_prediction = 1
learning_rate = 0.0001  # подбираем необходимую скорость обучения

for i in range(30):
    prediction = neural_networks(inp, weights)
    error = get_error(true_prediction, prediction)
    print("Prediction: %.10f, Weights: %s, Error: %.20f" % (prediction, weights, error))
    delta = (prediction - true_prediction) * inp * learning_rate
    weights = weights - delta


inp = np.array([150, 40])
weights = np.array([0.2, 0.3])
true_prediction = 1
learning_rate = 0.000001  # подбираем необходимую скорость обучения

for i in range(30):
    prediction = neural_networks(inp, weights)
    error = get_error(true_prediction, prediction)
    print("Prediction: %.10f, Weights: %s, Error: %.20f" % (prediction, weights, error))
    delta = (prediction - true_prediction) * inp * learning_rate
    weights = weights - delta


inp = np.array([150, 40])
weights = np.array([0.2, 0.3])
true_prediction = 1
learning_rate = 0.00001

for i in range(1000):
    prediction = neural_networks(inp, weights)
    error = get_error(true_prediction, prediction)
    print("Prediction: %.10f, Weights: %s, Error: %.20f" % (prediction, weights, error))
    delta = (prediction - true_prediction) * inp * learning_rate
    weights = weights - delta


# При успешном обучении увеличение количества итераций повышает предсказания до минимальной погрешности
# Перестает давать улучшения, когда точность почти максимальная или идеальная
