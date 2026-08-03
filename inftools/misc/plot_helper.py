""" Helper functions for plotting  """

def show_or_save(save: str = "") -> None:

    import matplotlib.pyplot as plt

    if save:
        plt.savefig(save)
        print(f"[INFO] Figure saved to {save}")
    else:
        plt.show()


SAVE_HELP = (
    "Save the figure to this filename (e.g. plot.png) instead of "
    "calling plt.show()."
)
