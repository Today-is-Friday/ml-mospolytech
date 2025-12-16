import numpy as np


def neural_networks(inp, weights):
    return inp.dot(weights)


def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


# def get_error(true_prediction, predction):
#     return np.sqrt(np.mean((true_prediction - predction) ** 2))


# Объявим несколько наборов данных на два входа
inp = np.array(
    [
        [150, 40],
        [140, 35],
        [155, 45],
        [185, 95],
        [145, 40],
        [195, 100],
        [180, 95],
        [170, 80],
        [160, 90],
    ]
)
weights = np.array([0.2, 0.3])
true_predictions = np.array([0, 0, 0, 100, 0, 100, 100, 100, 100])
learning_rate = 0.000001

for i in range(500):
    error = 0
    delta = 0
    for j in range(len(inp)):
        current_inp = inp[j]
        true_prediction = true_predictions[j]
        prediction = neural_networks(current_inp, weights)
        error += get_error(true_prediction, prediction)
        # print(
        #     "Prediction: %.10f, True_prediction: %.10f, Weights: %s"
        #     % (prediction, true_prediction, weights)
        # )
        delta += (prediction - true_prediction) * current_inp * learning_rate
        weights = weights - delta / len(inp)
        # print("Errors: %.10f" % error)
        # print("-------------------")


print(neural_networks(np.array([150, 45]), weights))
print(neural_networks(np.array([170, 85]), weights))

# более большой learning_rate может ускорить обучение, но появляется вероятность overflow encountered in scalar power
# более маленький learning_rate делает результат менее точным
# при маленьних колическах эпох нейронка дает значения близкие к 50
# 500+ эпох не дают ощутимой разницы
# RMSE никак не повлиял на обучение.
