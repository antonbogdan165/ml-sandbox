import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from vis import error_visualisation, universal_visualisation


def prepare_data(file_path):
    data = pd.read_csv(file_path, index_col=0)

    x = data[["x_1", "x_2"]].values
    y = data["y"]
    categorized_y = to_categorical(y, 2)
    return x, categorized_y, y.values


def train_model(x_train, y_train, x_val, y_val):
    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=(2,)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=50, restore_best_weights=True
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=1000,
        batch_size=10,
        callbacks=[early_stop],
        verbose=0,
    )
    return model, history


if __name__ == "__main__":
    X, y, y_raw = prepare_data("one_circle.csv")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_raw
    )

    model, history = train_model(X_train, y_train, X_val, y_val)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))

    error_visualisation(history.history["loss"], ax_right)

    universal_visualisation(
        X, y_raw, ax_left, lambda grid: model.predict(grid, verbose=0)[:, 1]
    )

    plt.tight_layout()
    plt.show()
