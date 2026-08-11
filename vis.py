import numpy as np


def universal_visualisation(X, y, ax_left, predict_func=None, title=None):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    if predict_func is not None:
        x_range = np.arange(x_min, x_max, 0.02)
        y_range = np.arange(y_min, y_max, 0.02)

        xx, yy = np.meshgrid(x_range, y_range)

        grid_points = np.c_[xx.ravel(), yy.ravel()]

        predictions = np.array(predict_func(grid_points)).reshape(xx.shape)

        ax_left.contourf(xx, yy, predictions, levels=20, cmap="coolwarm", alpha=0.3)
        try:
            ax_left.contour(
                xx,
                yy,
                predictions,
                levels=[0.5],
                colors="#333333",
                linewidths=1.5,
                linestyles="--",
            )
        except UserWarning:
            pass

    ax_left.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="coolwarm",
        edgecolor="white",
        linewidth=0.8,
        s=60,
        alpha=0.9,
    )

    if title:
        ax_left.set_title(title, fontsize=11, fontweight="bold", pad=8)

    ax_left.set_xlim(x_min, x_max)
    ax_left.set_ylim(y_min, y_max)
    ax_left.grid(True, linestyle=":", alpha=0.4, color="gray")
    ax_left.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax_left.spines[spine].set_visible(False)


def error_visualisation(errors, ax_right, title=None):
    epochs = np.arange(1, len(errors) + 1)

    ax_right.plot(epochs, errors, color="#2b5c8f", linewidth=2)
    ax_right.fill_between(epochs, errors, alpha=0.15, color="#2b5c8f")

    if title:
        ax_right.set_title(title, fontsize=11, fontweight="bolt", pad=8)

    ax_right.set_xlabel("Epoch", fontsize=9)
    ax_right.set_ylabel("Loss", fontsize=9)

    ax_right.grid(True, linestyle=":", alpha=0.4, color="gray")
    ax_right.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax_right.spines[spine].set_visible(False)
