import numpy as np


def relu(x):
    return (x > 0) * x


def reluderiv(x):
    return x > 0


np.random.seed(0)

inp = np.array([[15, 10], [15, 15], [15, 20], [25, 10]])
true_prediction = np.array([[10, 20, 15, 20]]).T

layer_hid_size = 8
layer_in_size = len(inp[0])
layer_out_size = len(true_prediction[0])

weights_hid = 2 * np.random.random((layer_in_size, layer_hid_size)) - 1
weights_hid2 = 2 * np.random.random((layer_hid_size, layer_hid_size)) - 1
weights_out = 2 * np.random.random((layer_hid_size, layer_out_size)) - 1

learning_rate = 0.00001  # задаем скорость обучения
num_epoch = 600  # установим количество эпох

for i in range(num_epoch):
    layer_out_error = 0  # задаем значение ошибки для вычисления дельты
    for i in range(len(inp)):
        layer_in = inp[i : i + 1]
        layer_hid = relu(np.dot(layer_in, weights_hid))
        layer_out = layer_hid.dot(weights_out)
        layer_out_error += np.sum(layer_out - true_prediction[i : i + 1]) ** 2
        layer_out_delta = true_prediction[i : i + 1] - layer_out
        layer_hid_delta = layer_out_delta.dot(weights_out.T) * reluderiv(layer_hid)
        weights_out += learning_rate * layer_hid.T.dot(layer_out_delta)
        weights_hid += learning_rate * layer_in.T.dot(layer_hid_delta)
        print(
            "Predictions: %s, True predictions: %s"
            % (layer_out, true_prediction[i : i + 1])
        )
        print("Errors: %.4f" % layer_out_error)


# Большее количество нейронов полижетельно влияет на качество предикта (в разумных количествах). Но при этом замедляет обучение.
# 32 нейрона приводят к ошибке примерно 0.0405. При 8 32.0993
# learning rate может как ускорить обучение так и замедлить. Но при больших значениях может полностью убить обучение
# при этом всем не обязателен overflow,  многие нейроны могут умереть и всегда показывать 0
# так же такая аномалия может произойти из-за плохих стартовых весов
# количество эпох является одним из ключей к хорошему обучению. Большее количество эпох улучшит нейросеть, но начиная с некоторой величчины обучение доходит до максимума
