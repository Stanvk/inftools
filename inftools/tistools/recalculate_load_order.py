from typing import Annotated, List, Optional
import typer


def recalculate_load_order(
    load_dir: Annotated[str, typer.Option("-load", help="path to the load/ directory containing numbered path folders")] = "load",
    toml: Annotated[str, typer.Option("-toml", help="infretis toml file defining the orderparameter")] = "infretis.toml",
    out_name: Annotated[str, typer.Option("-out", help="name of the recalculated file written next to each order.txt")] = "order_recalculated.txt",
    top: Annotated[str, typer.Option("-top", help="topology file passed to MDAnalysis (optional, depends on format)")] = "",
    mda_format: Annotated[str, typer.Option("-format", help="trajectory format for MDAnalysis (e.g. 'xyz', 'xtc', 'trr'); auto-detected from file extension when empty")] = "",
    include_rejected: Annotated[bool, typer.Option("-rejected/-no-rejected", help="also process rejected/<k>/ subfolders")] = True,
    overwrite: Annotated[bool, typer.Option("-overwrite/-no-overwrite", help="overwrite existing recalculated files")] = False,
    paths: Annotated[Optional[List[int]], typer.Option("-paths", help="only process these path numbers (e.g. -paths 1 -paths 2). Empty = all.")] = None,
):
    """Recalculate the orderparameter for every path in a load/ directory and
    write the result to a new file (default: order_recalculated.txt) next to
    each existing order.txt.

    Handles both:

      - accepted paths:  <load>/<pn>/order.txt              (trajs in <pn>/accepted/)
      - rejected paths:  <load>/<pn>/rejected/<k>/order.txt (trajs in that same folder)

    The output file mirrors the OrderPathFormatter layout (leading '# Cycle:'
    comment + column header + whitespace-separated rows: step, main-orderp, cv1, ...).
    """

    import os

    import numpy as np
    import tomli
    import MDAnalysis as mda

    from infretis.classes.orderparameter import create_orderparameter
    from infretis.classes.system import System

    with open(toml, "rb") as toml_file:
        toml_dict = tomli.load(toml_file)
    orderparameter = create_orderparameter(toml_dict)

    cp2k_box = None
    if toml_dict.get("engine", {}).get("class", "").lower() == "cp2k":
        from infretis.classes.engines.cp2k import read_cp2k_box
        cp2k_inp = os.path.join(toml_dict["engine"]["input_path"], "cp2k.inp")
        box, _ = read_cp2k_box(cp2k_inp)
        cp2k_box = list(box) + [90.0] * 3

    if not os.path.isdir(load_dir):
        raise NotADirectoryError(f"{load_dir} is not a directory")

    pn_filter = set(paths) if paths else None

    def _iter_targets():
        """Yield (pdir, traj_subdir, status) for each order.txt to (re)compute."""
        for entry in sorted(os.listdir(load_dir), key=lambda x: (not x.isdigit(), x)):
            pdir = os.path.join(load_dir, entry)
            if not os.path.isdir(pdir) or not entry.isdigit():
                continue
            if pn_filter is not None and int(entry) not in pn_filter:
                continue

            if os.path.isfile(os.path.join(pdir, "traj.txt")):
                yield pdir, os.path.join(pdir, "accepted"), "ACC"

            if include_rejected:
                rej_root = os.path.join(pdir, "rejected")
                if os.path.isdir(rej_root):
                    for rk in sorted(os.listdir(rej_root), key=lambda x: (len(x), x)):
                        rdir = os.path.join(rej_root, rk)
                        if not os.path.isdir(rdir):
                            continue
                        if os.path.isfile(os.path.join(rdir, "traj.txt")):
                            yield rdir, rdir, "REJ"

    def _recalc_one(pdir: str, traj_subdir: str, status: str) -> str:
        out_file = os.path.join(pdir, out_name)
        if os.path.isfile(out_file) and not overwrite:
            return f"skip (exists): {out_file}"

        traj_txt = os.path.join(pdir, "traj.txt")
        tdata = np.loadtxt(traj_txt, dtype=str, ndmin=2)
        if tdata.size == 0:
            return f"skip (empty traj.txt): {traj_txt}"

        files = sorted(set(tdata[:, 1]))
        fmt = mda_format or files[0].split(".")[-1]

        unis = {}
        for f in files:
            fpath = os.path.join(traj_subdir, f)
            if not os.path.isfile(fpath):
                raise FileNotFoundError(f"missing trajectory frame: {fpath}")
            unis[f] = (
                mda.Universe(top, fpath, format=fmt)
                if top
                else mda.Universe(fpath, format=fmt)
            )
            if cp2k_box is not None:
                unis[f].dimensions = cp2k_box

        with open(out_file, "w") as fh:
            fh.write(f"# Cycle: 0, status: {status}, move: recalc\n")
            fh.write("#       Time       Orderp\n")
            for row in tdata:
                step, fname, index, _vel = row[0], row[1], row[2], row[3]
                ts = unis[fname].trajectory[int(index)]
                system = System()
                system.config = (fname, int(index))
                system.pos = ts.positions
                system.box = ts.dimensions[:3] if ts.dimensions is not None else None
                op = orderparameter.calculate(system)
                cols = ["{:>10d}".format(int(step))] + [
                    "{:>12.6f}".format(float(v)) for v in op
                ]
                fh.write(" ".join(cols) + "\n")

        return f"wrote ({status}): {out_file}"

    n_ok = n_skip = n_err = 0
    for pdir, traj_subdir, status in _iter_targets():
        try:
            msg = _recalc_one(pdir, traj_subdir, status)
            print(f"[ INFO ] {msg}")
            if msg.startswith("skip"):
                n_skip += 1
            else:
                n_ok += 1
        except Exception as exc:
            n_err += 1
            print(f"[ ERROR ] {pdir}: {exc}")

    print(f"[ DONE ] wrote={n_ok} skipped={n_skip} errors={n_err}")
