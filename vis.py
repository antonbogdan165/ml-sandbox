import numpy as np


def universal_visualisation(X, y, predict_func, ax_left):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    x_range = np.arange(x_min, x_max, 0.02)
    y_range = np.arange(y_min, y_max, 0.02)

    xx, yy = np.meshgrid(x_range, y_range)

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    predictions = predict_func(grid_points)
    predictions = predictions.reshape(xx.shape)

    ax_left.contourf(xx, yy, predictions, alpha=0.2, cmap="bwr")
    ax_left.contour(xx, yy, predictions, colors="black", linewidths=2, levels=[0.5])
    ax_left.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", s=100, edgecolors="k")
    ax_left.grid(True, linestyle="--", alpha=0.5)
    ax_left.set_xlim(x_min, x_max)
    ax_left.set_ylim(y_min, y_max)


def error_visualisation(errors, ax_right):
    ax_right.plot(errors, color="black", linewidth=2)
    ax_right.grid(True)
