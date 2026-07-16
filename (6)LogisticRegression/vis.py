import matplotlib.pyplot as plt
import numpy as np


def universal_visualisation(features, labels, predict_func, ax_bot):
    x_min, x_max = features[:, 0].min() - 0.5, features[:, 0].max() + 0.5
    y_min, y_max = features[:, 1].min() - 0.5, features[:, 1].max() + 0.5

    x_range = np.arange(x_min, x_max, 0.02)
    y_range = np.arange(y_min, y_max, 0.02)

    xx, yy = np.meshgrid(x_range, y_range)

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    predictions = predict_func(grid_points)
    predictions = predictions.reshape(xx.shape)

    ax_bot.contourf(xx, yy, predictions, alpha=0.2, cmap="bwr")
    ax_bot.contour(xx, yy, predictions, colors="black", linewidths=2, levels=[0.5])
    ax_bot.scatter(
        features[:, 0], features[:, 1], c=labels, cmap="bwr", s=100, edgecolors="k"
    )
    ax_bot.grid(True, linestyle="--", alpha=0.5)
    ax_bot.set_xlim(x_min, x_max)
    ax_bot.set_ylim(y_min, y_max)


def error_visualisation(errors, ax_top):
    ax_top.plot(errors, color="black", linewidth=2)
    ax_top.grid(True)
