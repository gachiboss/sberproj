import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

"""
plt.figure(figsize=(10, 5))
for i in range(15):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
plt.show()
"""

model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)), #1 - 1 байт
    Dense(128, activation='relu'),
    Dense(10, activation='softmax'),
])
print(model.summary())

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train_cat, batch_size=32, epochs=8, validation_split=0.2) #batch - на 32 трен появляется тест

model.evaluate(x_test, y_test_cat)

n = 0
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print(res)
print(f"Это цифра {np.argmax(res)}")

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()