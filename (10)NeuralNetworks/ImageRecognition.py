import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def train_model(x_train, y_train_cat, x_test, y_test_cat):
    model = Sequential(
        [
            Dense(256, input_shape=(784,)),
            BatchNormalization(),
            Activation("relu"),
            Dropout(0.3),
            Dense(128),
            BatchNormalization(),
            Activation("relu"),
            Dropout(0.3),
            Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    my_callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-5),
    ]

    model.fit(
        x_train,
        y_train_cat,
        epochs=20,
        batch_size=64,
        validation_data=(x_test, y_test_cat),
        callbacks=my_callbacks,
        verbose=1,
    )
    return model


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train_reshaped = x_train.reshape(-1, 28 * 28) / 255.0
    x_test_reshaped = x_test.reshape(-1, 28 * 28) / 255.0

    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    model = train_model(x_train_reshaped, y_train_cat, x_test_reshaped, y_test_cat)

    test_loss, test_acc = model.evaluate(x_test_reshaped, y_test_cat, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    all_predictions_vector = model.predict(x_test_reshaped)
    all_predictions = np.argmax(all_predictions_vector, axis=1)

    correct_indices = np.where(all_predictions == y_test)[0]
    incorrect_indices = np.where(all_predictions != y_test)[0]

    num_incorrect = min(2, len(incorrect_indices))
    num_correct = 10 - num_incorrect

    if num_incorrect > 0:
        sample_incorrect = np.random.choice(
            incorrect_indices, num_incorrect, replace=False
        )
    else:
        sample_incorrect = np.array([], dtype=int)

    sample_correct = np.random.choice(correct_indices, num_correct, replace=False)

    indices_to_plot = np.concatenate([sample_correct, sample_incorrect])
    np.random.shuffle(indices_to_plot)

    plt.figure(figsize=(15, 4))
    for i, idx in enumerate(indices_to_plot):
        plt.subplot(1, 10, i + 1)
        plt.imshow(x_test[idx], cmap="gray_r")

        true_label = y_test[idx]
        pred_label = all_predictions[idx]

        color = "green" if true_label == pred_label else "red"
        plt.title(f"Label: {true_label}\nPrediction: {pred_label}", color=color)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
