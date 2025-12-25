from keras.src.datasets import mnist
import numpy as np


def relu(x):
    return (x > 0) * x


def reluderiv(x):
    return x > 0


train_images_count = 1000
test_images_count = 10000
pixels_per_image = 28 * 28
digits_num = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_images = (
    x_train[0:train_images_count].reshape(train_images_count, pixels_per_image) / 255
)
train_labels = y_train[0:train_images_count]
print(train_labels)

test_images = (
    x_test[0:test_images_count].reshape(test_images_count, pixels_per_image) / 255
)
test_labels = y_test[0:test_images_count]


one_hot_labels = np.zeros((len(train_labels), digits_num))
for j in range(len(train_labels)):
    one_hot_labels[j][train_labels[j]] = 1
train_labels = one_hot_labels

one_hot_labels = np.zeros((len(test_labels), digits_num))

for i, j in enumerate(test_labels):
    one_hot_labels[i][j] = 1
test_labels = one_hot_labels

np.random.seed(2)

hidden_size = 50
weight_hid = 0.2 * np.random.random((pixels_per_image, hidden_size)) - 0.1
weight_out = 0.2 * np.random.random((hidden_size, digits_num)) - 0.1

learning_rate = 0.01
dropout_rate = 0.2
num_epoch = 100
batch_size = 50


for epoch in range(num_epoch):
    correct_answers = 0

    for j in range(train_images_count // batch_size):
        batch_start = j * batch_size
        batch_end = batch_start + batch_size

        layer_in = train_images[batch_start:batch_end]

        layer_hid_raw = layer_in.dot(weight_hid)
        layer_hid = relu(layer_hid_raw)

        dropout_mask = np.random.rand(*layer_hid.shape) > dropout_rate
        layer_hid *= dropout_mask
        layer_hid /= 1 - dropout_rate

        layer_out = layer_hid.dot(weight_out)

        for k in range(batch_size):
            correct_answers += int(
                np.argmax(layer_out[k]) == np.argmax(train_labels[batch_start + k])
            )

        layer_out_delta = (layer_out - train_labels[batch_start:batch_end]) / batch_size

        layer_hid_delta = (
            layer_out_delta.dot(weight_out.T) * reluderiv(layer_hid_raw) * dropout_mask
        )

        weight_out -= learning_rate * layer_hid.T.dot(layer_out_delta)
        weight_hid -= learning_rate * layer_in.T.dot(layer_hid_delta)

    print(
        f"Epoch {epoch}: accuracy = "
        f"{correct_answers * 100 / train_images_count:.2f}%"
    )


print("Epoch: ", i)
print("Accuracy: %.2f" % (correct_answers * 100 / len(train_images)))

correct_answers = 0
for j in range(len(test_images)):
    layer_in = test_images[j : j + 1]
    layer_hid = relu(np.dot(layer_in, weight_hid))
    layer_out = np.dot(layer_hid, weight_out)
    correct_answers += int(np.argmax(layer_out) == np.argmax(test_labels[j : j + 1]))

print("Accuracy: %.2f" % (correct_answers * 100 / len(test_images)))
