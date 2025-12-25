import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidderiv(x):
    s = sigmoid(x)
    return s * (1 - s)


def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)


def predict(inp):
    global layer_hid, layer_out
    layer_hid = sigmoid(inp.dot(weight_hid))
    layer_out = softmax(layer_hid.dot(weight_out))

    return np.argmax(layer_out)


x = np.array(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ]
)


y = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]
)

input_size = len(x[0])
hidden_size = 15
output_size = len(y[0])
learning_rate = 0.01
epochs = 999

np.random.seed(3)

weight_hid = 0.2 * np.random.random((input_size, hidden_size)) - 0.1
weight_out = 0.2 * np.random.random((hidden_size, output_size)) - 0.1

layer_hid = np.array((input_size, hidden_size))
layer_out = np.array((hidden_size, output_size))

for epoch in range(epochs):
    for j in range(len(x)):
        inp = x[j : j + 1]
        target = y[j : j + 1]

        # forward
        hid_raw = inp.dot(weight_hid)
        layer_hid = sigmoid(hid_raw)
        out_raw = layer_hid.dot(weight_out)
        layer_out = softmax(out_raw)

        # backward
        layer_out_delta = layer_out - target
        layer_hid_delta = (
            layer_out_delta.dot(weight_out.T) * layer_hid * (1 - layer_hid)
        )

        weight_out -= learning_rate * layer_hid.T.dot(layer_out_delta)
        weight_hid -= learning_rate * inp.T.dot(layer_hid_delta)

        print(np.argmax(layer_out))


for i in range(len(x)):
    print(x[i], "→", predict(x[i : i + 1]), " expected:", np.argmax(y[i]))
