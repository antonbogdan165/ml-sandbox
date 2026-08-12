import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC

from vis import universal_visualisation


def prepare_data(file_path):
    data = pd.read_csv(file_path)
    features = data[["x_1", "x_2"]].to_numpy()
    labels = data["y"].to_numpy()
    return features, labels


def train_model(features, labels, C=(0.01, 1, 100)):
    models = {}
    for c in C:
        model = SVC(kernel="linear", C=c)
        model.fit(features, labels)
        models[c] = model
    return models


if __name__ == "__main__":
    features, labels = prepare_data("linear.csv")
    models = train_model(features, labels)

    fig, axes = plt.subplots(1, len(models), figsize=(15, 5))
    for ax, (c, model) in zip(axes, models.items()):
        universal_visualisation(features, labels, ax, model.predict, f"C={c}")

    plt.tight_layout()
    plt.show()
