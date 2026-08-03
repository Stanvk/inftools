from typing import Annotated as Atd
from typer import Option as Opt

def plot_hist(
    xval: Atd[str, Opt(help="xval histogram file")] = "histo_xval.txt",
    yval: Atd[str, Opt(help="yval histogram file")] = "histo_yval.txt",
    hist: Atd[str, Opt(help = "the histogram or free-energy file")] = "histo_free_energy.txt",
    save: Atd[str, Opt("-save", help="Save the figure to this filename (e.g. plot.png) instead of calling plt.show(). Useful on headless compute nodes.")] = "",
    ):
    """
    Plot the output from the WHAM free-energy analysis.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    from inftools.misc.plot_helper import show_or_save

    x = np.loadtxt(xval)
    y = np.loadtxt(yval)
    h = np.loadtxt(hist)
    plt.pcolormesh(x, y, h.T)
    show_or_save(save)
