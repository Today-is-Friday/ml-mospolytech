def neural_networks(inp, weight):
    return inp * weight


def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2


def grad():
    global weight, inp, learning_rate
    for i in range(100):  # обучать будем на протяжении 10 итераций (эпох)
        prediction = neural_networks(inp, weight)
        error = get_error(true_prediction, prediction)
        # print(
        #     "Prediction: %.10f, Weight: %.5f, Error: %.20f"
        #     % (prediction, weight, error)
        # )
        delta = (
            (prediction - true_prediction) * inp * learning_rate
        )  # это производная, она же градиент

        weight = weight - delta

    print(prediction, "\n")


inp = 30
weight = 0.2
true_prediction = 70
learning_rate = 0.001  # alpha-коэффициент
grad()


inp = 30
weight = 0.2
true_prediction = 70
learning_rate = 0.01
# grad()

inp = 30
weight = 0.2
true_prediction = 70
learning_rate = 0.1
# grad()

inp = 30
weight = 0.2
true_prediction = 70
learning_rate = 0.0001
grad()

# Самое лучшее 0.001
# При > 0.001 обучение ломается и предсказания уходят в бесконечность
# При < 0.001 обучение медленное

# При 100 итераций rate = 0.1 вызывает OverflowError: (34, 'Numerical result out of range')
# rate = 0.0001 ближе к сотой итерации обучилась, а rate = 0.01 идет так же в бесконечность
