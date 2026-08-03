from typing import Annotated
import typer

def check_data(
    data: Annotated[str, typer.Option("-data")] = "infretis_data.txt",
    toml: Annotated[str, typer.Option("-toml")] = "infretis.toml",
    plot: Annotated[bool, typer.Option("-plot")] = False,
    save: Annotated[str, typer.Option("-save", help="Save the figure to this filename (e.g. plot.png) instead of calling plt.show(). Useful on headless compute nodes.")] = "",
    ):
    """Check that all ensembles [i] contain paths that cross the interface of the next ensemble [i+1]. Can be used to plot the maximum OP value of all paths within each ensemble."""
    import numpy as np
    import matplotlib.pyplot as plt
    import tomli

    from inftools.misc.plot_helper import show_or_save

    with open(toml, "rb") as toml_file:
        toml_dict = tomli.load(toml_file)

    interfaces = toml_dict["simulation"]["interfaces"][:]
    x = np.loadtxt(data, dtype = str)

    N_ens = len(interfaces)

    # array where element [i,j] is the maximum OP
    # value of line i in ensemble j.
    OP_matrix = np.zeros((x.shape[0],N_ens))
    for i, xi in enumerate(x):
        x_tmp = x[i, 3:N_ens + 3]
        idx = np.where(x_tmp!="----")
        # drop load paths which are all '----'
        if len(idx[0])==0:
            continue

        OP_matrix[i, idx[0]] = xi[2].astype(float)

    if plot:
        f,a = plt.subplots(1, N_ens, sharey = True)

    for ens in range(N_ens):

        ens_intf = ens
        #[0-] ens stuff
        if ens > 0:
            ens_intf -=1
        # Left and Middle interfaces
        L = interfaces[ens_intf]
        M = interfaces[ens_intf + 1]

        path_nr = OP_matrix[:,ens]!=0
        # max OP value of all paths in ensemble ens
        op = OP_matrix[:,ens][path_nr]

        if plot:
            a[ens].axvline(L, c = 'k', lw=3)
            a[ens].axvline(M, c='k', lw=3)
            a[ens].scatter(op,x[path_nr, 0].astype(int))
            if op.shape[0] > 0:
                xlim_hi = max(max(op), M)
            else:
                xlim_hi = M
            a[ens].set(xlim=(L, xlim_hi), xlabel = 'maximum OP', ylabel = 'path nr. in infretis_data.txt', title = f"ensemble: {ens}, intf={M}")
        #print(ens, op)
        if len(op)==0:
            print(f"Ensemble {ens} does not contain any paths.")
        elif ens > 0 and max(op) < M:
            print(f"Ensemble {ens} is missing paths that cross interface {ens_intf+1} with value {M}")

    if plot:
        show_or_save(save)
