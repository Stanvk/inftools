import os
from typing import Annotated

import typer
from inftools.misc.infinit_helper import *

# export _TYPER_STANDARD_TRACEBACK=1

"""Drop-in addition for inftools/tistools/initial_paths.py.

Paste the function below into that file (it needs the `read_toml` that is
already star-imported there from inftools.misc.infinit_helper). No registration
is needed -- get_mapper() picks up every function defined in tistools/*.py.

Do NOT add module-level helper functions to that file: they would each become
an `inft` subcommand. This is why everything here lives inside the function.

The imports at the top of this file mirror initial_paths.py so the module can
be exercised standalone before dropping it in.
"""
import os
from typing import Annotated

import typer
from inftools.misc.infinit_helper import *


def initial_paths_from_reverse(
    path: Annotated[str, typer.Option("-path", help="Directory of a reactive path from the forward simulation, e.g. 'load/5627'.")],
    toml: Annotated[str, typer.Option("-toml", help="The .toml file of the reverse simulation. 'interfaces' and 'load_dir' are read from it.")] = "infretis.toml",
    shift: Annotated[float, typer.Option("-shift", help="Mirror constant of the order parameter, op_rev = shift - op. Defaults to interfaces[0] + interfaces[-1].")] = None,
    ens: Annotated[str, typer.Option("-ens", help="Ensembles to seed, given as 'first:last' (inclusive). Defaults to every plus ensemble, leaving [0-] untouched.")] = "",
    keep_op: Annotated[bool, typer.Option(help="Append the forward order parameter as the last column of order.txt, so both simulations can be histogrammed on the same collective variable.")] = True,
    copy: Annotated[bool, typer.Option(help="Copy the trajectory frames into every ensemble directory instead of symlinking them. They are usually far too large to copy.")] = False,
    ):
    """Seed a reverse (B->A) simulation with a reactive path from the forward

    (A->B) simulation. A reactive path spans the whole barrier, so time-reversed
    it is a valid initial path in *every* plus ensemble of the reverse
    simulation at once. This skips the barrier-climbing phase of 'infinit'.

    Only the bookkeeping files are touched: infretis stores velocity reversal as
    a per-frame flag in the 'vel' column of traj.txt, so the trajectory files
    are reused as they are. The order parameter is remapped analytically as
    op_rev = shift - op, which must agree with the mirrored orderparameter class
    the reverse simulation is configured with.

    The [0-] ensemble is not seeded. It needs a path that dips into the reverse
    reactant state, which a reactive path never does -- run
    'inft generate_zero_paths' for that one."""
    import pathlib as pl
    import shutil

    import numpy as np

    config = read_toml(toml)
    intfs = config["simulation"]["interfaces"]
    load_dir = pl.Path(config["simulation"].get("load_dir", "load"))
    src = pl.Path(path)

    if shift is None:
        shift = intfs[0] + intfs[-1]
        print(f"Mirroring the order parameter with shift = {shift}")

    if ens:
        first, last = (int(i) for i in ens.split(":"))
    else:
        # ensemble i uses interfaces[i-1] as its middle interface, so the plus
        # ensembles are 1 ... len(interfaces)-1
        first, last = 1, len(intfs) - 1

    # read the forward path and drop the step column of each file. traj.txt is
    # read by hand because np.loadtxt warns on its comment lines when dtype=str
    order = np.loadtxt(src / "order.txt", ndmin=2)[::-1, 1:]
    traj = np.array(
        [
            ln.split()
            for ln in (src / "traj.txt").read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ],
        dtype=str,
    )[::-1, 1:]
    energy = None
    if (src / "energy.txt").is_file():
        energy = np.loadtxt(src / "energy.txt", ndmin=2)[::-1, 1:]
        if len(energy) != len(order):
            raise ValueError(
                f"energy.txt has {len(energy)} frames but order.txt has "
                f"{len(order)} in {src}"
            )
    if len(traj) != len(order):
        raise ValueError(
            f"traj.txt has {len(traj)} frames but order.txt has "
            f"{len(order)} in {src}"
        )

    # mirror the order parameter, keeping the forward value as the last column
    op = shift - order[:, 0]
    if keep_op:
        order = np.column_stack((op, order[:, 1:], order[:, 0]))
    else:
        order = np.column_stack((op, order[:, 1:]))

    # the reversed path has to start inside the reverse reactant state and end
    # in the product state, otherwise it is not valid in the plus ensembles.
    # infretis does not check initial paths on load, so we check here
    if not op[0] < intfs[0]:
        raise ValueError(
            f"The reversed path starts at {op[0]:.5f}, which is not below "
            f"interfaces[0] = {intfs[0]}. Check '-shift' and that {src} is a "
            "reactive path."
        )
    if not op[-1] > intfs[-1]:
        raise ValueError(
            f"The reversed path ends at {op[-1]:.5f}, which does not reach "
            f"interfaces[-1] = {intfs[-1]}. Only a reactive path can seed all "
            "plus ensembles."
        )
    print(
        f"Reversed path: {len(op)} frames, "
        f"{op[0]:.5f} -> {op[-1]:.5f}, crossing all {len(intfs)} interfaces"
    )

    # infretis stores velocity reversal per frame, so reversing the path is
    # just flipping this column
    fnames, idxs, vels = traj[:, 0], traj[:, 1], -traj[:, 2].astype(int)
    steps = np.arange(len(order)).reshape(-1, 1)

    for i in range(first, last + 1):
        dirname = load_dir / str(i)
        accepted = dirname / "accepted"
        accepted.mkdir(parents=True, exist_ok=True)

        np.savetxt(
            str(dirname / "order.txt"),
            np.hstack((steps, order)),
            fmt=["%d"] + ["%12.6f"] * order.shape[1],
        )
        np.savetxt(
            str(dirname / "traj.txt"),
            np.c_[[str(j) for j in range(len(traj))], fnames, idxs, vels],
            header=f"{'time':>10} {'trajfile':>15} {'index':>10} {'vel':>5}",
            fmt=["%10s", "%15s", "%10s", "%5s"],
        )
        if energy is not None:
            np.savetxt(
                str(dirname / "energy.txt"),
                np.hstack((steps, energy)),
                fmt=["%d"] + ["%14.6f"] * energy.shape[1],
            )

        for trajfile in np.unique(fnames):
            dest = accepted / trajfile
            if dest.exists() or dest.is_symlink():
                continue
            if copy:
                shutil.copy(src / "accepted" / trajfile, dest)
            else:
                dest.symlink_to((src / "accepted" / trajfile).resolve())

    n = last - first + 1
    how = "copied" if copy else "symlinked"
    print(f"Seeded ensembles {first}-{last} ({n} dirs) in {load_dir}, frames {how}")
    print(f"Ensemble 0 ([0-]) was not seeded, use 'inft generate_zero_paths'")


def generate_zero_paths(
    conf: Annotated[str, typer.Option("-conf", help="The name (not the path) of the initial configuration to propagate from. inftools will look in the input folder specified in the .toml file.")],
    toml: Annotated[str, typer.Option("-toml",)] = "infretis.toml",
    config : Annotated[str, typer.Option(hidden = True)] = None,
    ):
    """Generate initial paths for the [0-] and [0+]

    ensembles by propagating a single configuration forward
    and backward in time until it crosses the lambda0 interface.
    These can be used to e.g. push the system up the barrier using
    multiple infRETIS simulations."""
    import pathlib as pl

    import numpy as np
    from infretis.classes.engines.factory import create_engines
    from infretis.classes.orderparameter import create_orderparameters
    from infretis.classes.path import Path, paste_paths
    from infretis.classes.repex import REPEX_state
    from infretis.classes.system import System
    from infretis.setup import setup_config


    config0 = read_toml(toml)

    # make a directory we work from
    if config0["engine"]["class"] == "ase_external":
        tmp_dir = pl.Path("worker0")
        tmp_dir.mkdir(exist_ok = True)
    else:
        tmp_dir = pl.Path("worker0")
        tmp_dir.mkdir(exist_ok = False)
    load_dir = pl.Path(config0["simulation"].get("load_dir", "load"))
    load_dir.mkdir(exist_ok = False)

    initial_configuration = conf

    config0["runner"]["workers"] = 1
    write_toml(config0, "zero_paths.toml")
    # infretis parameters
    config = setup_config("zero_paths.toml")
    maxlen = config["simulation"]["tis_set"]["maxlength"]
    state = REPEX_state(config, minus=True)

    # setup ensembles
    state.initiate_ensembles()
    state.engines, state.engine_occ = create_engines(config)
    create_orderparameters(state.engines, config)

    # initial configuration to start from
    system0 = System()
    engine_key = list(state.engines.keys())[0]
    engine = state.engines[engine_key][0]
    engine.exe_dir = str(tmp_dir.resolve())
    wmdrun = False
    if "dask" in config.keys():
        wmdrun = config["dask"]["wmdrun"][0]
    elif "wmdrun" in config["runner"].keys():
        wmdrun = config["runner"]["wmdrun"][0]
    if wmdrun:
        engine.set_mdrun(
            {"wmdrun": wmdrun, "exe_dir": engine.exe_dir}
        )
    system0.set_pos((os.path.abspath(initial_configuration), 0))
    system0.order = engine.calculate_order(system0)
    engine.rgen = np.random.default_rng()
    engine.modify_velocities(system0, config["simulation"]["tis_set"])

    # empty paths we will fill forwards in time in [0-] and [0+]
    path0 = Path(maxlen=maxlen)
    path1 = Path(maxlen=maxlen)

    # propagate forwards from the intiial configuration
    # note that one of these does not get integrated because
    # the initial phasepoint is either below or above interface 0
    print("Propagating in ensemble [0-]")
    status0, message0 = engine.propagate(path0, state.ensembles[0], system0)
    system0.set_pos((os.path.abspath(initial_configuration), 0))
    system0.order = path0.phasepoints[0].order
    print("Propagating in ensemble [0+]")
    status1, message1 = engine.propagate(path1, state.ensembles[1], system0)

    # we did only one integration step in ensemble 0 because
    # we started above interface 0
    if path0.length == 1:
        print("Re-propagating [0-] since we started above lambda0")
        system0.set_pos((engine.dump_config(path1.phasepoints[-1].config), 0))
        system0.order = path1.phasepoints[-1].order
        path0 = Path(maxlen=maxlen)
        status0, message0 = engine.propagate(path0, state.ensembles[0], system0)

    # or we did only one integration step in ensemble 1 because
    # we started below interface 0
    elif path1.length == 1:
        print("Re-propagating [0+] since we started below lambda0")
        system0.set_pos((engine.dump_config(path0.phasepoints[-1].config), 0))
        system0.order = path0.phasepoints[-1].order
        path1 = Path(maxlen=maxlen)
        status1, message1 = engine.propagate(path1, state.ensembles[1], system0)

    else:
        raise ValueError("Something fishy!\
                Path lengths in one of the ensembles != 1")

    # backward paths
    path0r = Path(maxlen=maxlen)
    path1r = Path(maxlen=maxlen)

    print("Propagating [0-] in reverse")
    status0, message0 = engine.propagate(
        path0r, state.ensembles[0], path0.phasepoints[0], reverse=True
    )

    print("Propagating [0+] in reverse")
    status1, message1 = engine.propagate(
        path1r, state.ensembles[1], path1.phasepoints[0], reverse=True
    )

    print(f"Done! Making {load_dir} dir")
    # make load directories
    pathsf = [path0, path1]
    pathsr = [path0r, path1r]
    for i in range(2):
        dirname = load_dir / str(i)
        accepted = dirname / "accepted"
        orderfile = dirname / "order.txt"
        trajtxtfile = dirname / "traj.txt"
        dirname.mkdir()
        accepted.mkdir()
        # combine forward and backward path
        path = paste_paths(pathsr[i], pathsf[i])
        # save order paramter
        order = np.array([pp.order for pp in path.phasepoints])
        # return max op of [0+] path
        if i == 1:
            max_op = np.max(order[:,0])
        order = np.hstack((np.arange(len(order)).reshape(-1,1), np.array(order)))
        fmt = ["%d"] + ["%12.6f" for i in range(order.shape[1]-1)]
        np.savetxt(str(orderfile), order, fmt=fmt)
        N = len(order)
        # save traj.txt
        np.savetxt(
            str(trajtxtfile),
            np.c_[
                [str(i) for i in range(N)],
                [pp.config[0].split("/")[-1] for pp in path.phasepoints],
                [pp.config[1] for pp in path.phasepoints],
                [-1 if pp.vel_rev else 1 for pp in path.phasepoints],
            ],
            header=f"{'time':>10} {'trajfile':>15} {'index':>10} {'vel':>5}",
            fmt=["%10s", "%15s", "%10s", "%5s"],
        )
        # copy paths
        for trajfile in np.unique(
            [pp.config[0].split("/")[-1] for pp in path.phasepoints]
        ):
            src_path = tmp_dir / trajfile
            dest_path = accepted / trajfile
            shutil.move(src_path, dest_path)
    return max_op


def infinit(
    toml: Annotated[str, typer.Option("-toml", help="Path to .toml")] = "infretis.toml",
    log: Annotated[str, typer.Option("-log", help="File for logging output")] = "infretis_init.log",
    ):
    """The infretis initial path generator."""

    from inftools.exercises.puckering import initial_path_from_iretis

    # Based on the YouTube series:
    # https://www.youtube.com/watch?v=mW9tC2A7COs&list=PL5dSi5eZMe1iN_Uz8pTph6i8AGXhVUZIj&index=24

    # Lecture 04:
    # define the grid spacing for lambda values. All interface positions are
    # are rounded off to this vlue

    # skip this fraction of initial paths for analysis (when estimating new intf?)

    # compute efficiency measure for present set of interfaces and optimal set
    # (estimated from WHAM). If efficiency is worse than some factor, we update

    # estimated lower bound for local crossing probability

    # njumps Lp/Ls where Lp is average len full path, and Ls average len sub traj
    # lambda_cap to avoid A -> B trajs. E.g. 10% B->B paths in wf. Alternatively,
    # palce lambda_cap in half between lambdaN and lambda(N-1)

    # Lecture 05: set lambda0 lambdaN


    # TODO: restart, log file instead of print
    log = LightLogger(log)

    # we need among others parameters set in [infinit]
    config = read_toml(toml)
    # get the infinit settings from 'config' and set default parameters
    iset = set_default_infinit(config)

    if iset["cstep"]  == -1:
        log.log("Generating zero paths ...")
        init_conf = pl.Path(iset["initial_conf"]).resolve()
        max_op = generate_zero_paths(str(init_conf), toml = toml)
        log.log(f"Done with zero paths! Max op: {max_op}\n")
        iset["cstep"] = 0
        # for placing interfaces if we start with more than 1 worker
        intf = config["simulation"]["interfaces"]
        d_lambda = max_op - intf[0]
        nworkers = config["runner"]["workers"]
        lamres0 = d_lambda/nworkers
        # new interfaces to use for first infretis sim
        intf = [intf[0]] + [intf[0] + lamres0*(i+1) for i in range(nworkers-1)] + [intf[1]]
        sh_moves = ["sh", "sh"] + ["wf" for i in range(len(intf)-2)]

        # create symlink to load/1 path Nworker-1 times
        load_dir = pl.Path(config["simulation"].get("load_dir", "load"))
        load0 = load_dir / "1"
        for i in range(1,nworkers):
            loadn = load_dir / str(i + 1)
            #loadn.symlink_to(load0.relative_to(loadn.parent), target_is_directory=True)
            shutil.copytree(load0, loadn)

        c0 = read_toml(toml)
        c0["infinit"] = iset
        c0["simulation"]["interfaces"] = intf
        c0["simulation"]["shooting_moves"] = sh_moves
        write_toml(c0, "infretis.toml")

    log.log('Running infretis initialization "infinit" ...')
    if not pl.Path("infretis.toml").exists():
        print("Writing infretis.toml")
        c0 = read_toml(toml)
        c0["infinit"] = iset
        write_toml(c0, "infretis.toml")
    print_logo(step = -1)
    for iretis_steps in iset["steps_per_iter"][iset["cstep"]:]:
        log.log(f"Step {iset['cstep']}: Running infretis")
        success = run_infretis_ext(iretis_steps)
        if not success:
            print(f" *** infinit exiting loop at cstep={iset['cstep']}")
            return 1
        log.log("Updating interfaces.")
        update_toml_interfaces(config)
        msg = "interfaces = ["
        msg += ", ".join([str(intf) for intf in config["simulation"]["interfaces"]])
        msg += "]"
        log.log(msg)
        iset["cstep"] += 1
        update_toml(config)
        out = initial_path_from_iretis(
                config["simulation"].get("load_dir", "load"),
                "infretis.toml",
                restart = "restart.toml",
                return_pathnr = True)
        # update infretis.toml to be a restart.toml
        update_actives_toml(out)
        # rename restart file
        rename_file("restart.toml", f"restart_{iset['cstep']}.toml")
