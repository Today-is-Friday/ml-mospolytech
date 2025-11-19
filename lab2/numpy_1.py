import numpy as np

from random import randint

arr = np.array([1, 2, 3, 4, 5])
print(arr * 2)

arr1 = np.array(
    [
        [randint(0, 10), randint(0, 10), randint(0, 10)],
        [randint(0, 10), randint(0, 10), randint(0, 10)],
        [randint(0, 10), randint(0, 10), randint(0, 10)],
    ]
)

arr2 = np.array(
    [
        [randint(0, 10), randint(0, 10), randint(0, 10)],
        [randint(0, 10), randint(0, 10), randint(0, 10)],
        [randint(0, 10), randint(0, 10), randint(0, 10)],
    ]
)

print(arr1 * arr2)

arr3 = np.random.randint(0, 10, size=(10))

print(arr3)
print(arr3[1::2])

print(np.mean(arr3), np.std(arr3), np.max(arr3), np.min(arr3))
