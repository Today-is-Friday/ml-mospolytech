import numpy as np


def neural_networks(inp, weights):
    return inp.dot(weights)


# def get_error(true_prediction, prediction):
#     return (true_prediction - prediction) ** 2


def get_error(true_prediction, predction):
    return np.sqrt(np.mean((true_prediction - predction) ** 2))


inp = np.array([[150, 40], [170, 80], [160, 90]])

true_predictions = np.array([50, 120, 140])
weights = np.array([0.2, 0.3])
learning_rate = 0.00001  # подбираем необходимую скорость обучения

for i in range(100):
    error = 0
    for j in range(len(inp)):
        current_inp = inp[j]
        true_prediction = true_predictions[j]
        prediction = neural_networks(current_inp, weights)
        error += get_error(true_prediction, prediction)
        print(
            "Prediction: %.10f, True_prediction: %.10f, Weights: %s"
            % (prediction, true_prediction, weights)
        )
        delta = (prediction - true_prediction) * current_inp * learning_rate
        weights = weights - delta
        print("Errors: %.10f" % error)
        print("-------------------")

# если менять количество эпох, то изменится ошибка и веса. не всегда последняя эпоха будет
# меньше предыдущей, но на дистанции она будет уменьшаться до минимально возможной

# RMSE усредняет квадратичную ошибку и возвращает её в исходных единицах измерения
# Этот метод ошибки хорошо работает при примерно равном масштабе ошибок,
# но чувствительна к выбросам, так как одна большая ошибка может доминировать в значении метрики
