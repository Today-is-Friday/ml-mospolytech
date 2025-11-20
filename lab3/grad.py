import numpy as np


# реализуем простейшую нейронную сеть с одним входным и выходным нейронами
def neural_networks(inp, weight):
    return inp * weight


def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


inp = 0.9
weight = 0.2

true_prediction = 0.2
print(get_error(true_prediction, neural_networks(inp, weight)))


def grad():
    global weight, inp
    for i in range(1):  # обучать будем на протяжении 10 итераций (эпох)
        prediction = neural_networks(inp, weight)
        error = get_error(true_prediction, prediction)
        print(
            "Prediction: %.10f, Weight: %.5f, Error: %.20f"
            % (prediction, weight, error)
        )
        delta = (prediction - true_prediction) * inp  # это производная, она же градиент

        weight = weight - delta


grad()
p = neural_networks(inp, weight)
print(p)


inp = 1
weight = 0.01
grad()
p = neural_networks(inp, weight)
print(p)

inp = 1
weight = 2
true_prediction = 0.5
grad()
p = neural_networks(inp, weight)
print(p)
# inp * (weight - delta) ?= true_prediction


inp = -1
weight = 0.1
true_prediction = 0.8
grad()
p = neural_networks(inp, weight)
print(p)

# Если вес слишком большой и delta соответвенно тоже, то обучение невозможно
# Если вес слишком мал, то обучение будет долгим
# Только входные данные и разница между true_prediction и prediction
# Влияют на скорость, точность и возможность обучения
