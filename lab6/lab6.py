import numpy as np


def relu(x):
    return (x > 0) * x


def neural_networks(inp, weights):
    out = inp
    for w in weights[:-1]:
        out = relu(out.dot(w))

    out = out.dot(weights[-1])

    return out


inp = np.array([[15, 10], [15, 15], [15, 20], [25, 10]])
true_prediction = np.array([[10, 20, 15, 20]]).T

layer_hid_size = 8
layer_in_size = len(inp[0])
layer_out_size = len(true_prediction[0])

weights_hid = 2 * np.random.random((layer_in_size, layer_hid_size)) - 1
weights_hid2 = 2 * np.random.random((layer_hid_size, layer_hid_size)) - 1
weights_out = 2 * np.random.random((layer_hid_size, layer_out_size)) - 1

print(neural_networks(inp, [weights_hid, weights_hid2, weights_out]))
print(true_prediction)

# Большее количество нейронов позволит больше поразному взглянуть на вход,
# благодаря relu нейронка стала нелинейной
