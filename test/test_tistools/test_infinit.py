"""Test methods for doing TIS."""
import difflib
import filecmp
import os
import shutil
from pathlib import PosixPath
from subprocess import STDOUT, check_output

import pytest
import tomli
import tomli_w

from inftools.misc.infinit_helper import read_toml, refresh_datafile_maxop


@pytest.mark.heavy
def test_infinit_1(tmp_path: PosixPath) -> None:
    """Test infinit from phase point"""
    folder = tmp_path / "temp"
    folder.mkdir()
    basepath = PosixPath(__file__).parent
    load_dir = (
        basepath / "../../examples/turtlemd/double_well/load_copy"
    ).resolve()
    toml_dir = basepath / "data/infretis.toml"
    conf_dir = basepath / "data/initial.xyz"
    shutil.copy(str(toml_dir), str(folder))
    shutil.copy(str(conf_dir), str(folder))
    os.chdir(folder)

    # Run infinit
    os.system("inft infinit")

    # Check if files exist and that p is same or growing
    tomls = [f"infretis_{i}.toml" for i in range(2, 6)] + ["infretis.toml"]
    assert os.path.isfile(tomls[0])

    config0 = read_toml(tomls[0])
    assert "infinit" in config0
    for i,toml in enumerate(tomls):
        assert os.path.isfile(toml)
        assert os.path.isfile(f"combo_{i}.toml")
        assert os.path.isfile(f"combo_{i}.txt")
        # number of lines in combined infretis_data
        num_lines = num_lines = sum(1 for _ in open(f"combo_{i}.txt"))
        # check that we actually have data
        assert num_lines > 0
        # check that we have more data than previous run
        if i > 1:
            assert num_lines > prev_run_num_lines
        prev_run_num_lines = num_lines


def test_refresh_datafile_maxop(tmp_path: PosixPath) -> None:
    """Test that the data-file max-op column is refreshed from paths."""

    class DummyOrderFunction:
        def recalculate_order(self, order_vec):
            return [10.0 * float(order_vec[0]), *order_vec[1:]]

    load_dir = tmp_path / "load"
    load_dir.mkdir()

    for pnumber, orders in {
        1: [[1.0, 2.0], [5.0, 3.0]],
        2: [[2.0, 4.0], [4.0, 1.0]],
    }.items():
        pdir = load_dir / str(pnumber)
        accepted = pdir / "accepted"
        accepted.mkdir(parents=True)
        (accepted / "conf0.xyz").write_text("0\n")
        (accepted / "conf1.xyz").write_text("0\n")
        (pdir / "traj.txt").write_text(
            "\n".join(
                [
                    "0 conf0.xyz 0 1",
                    "1 conf1.xyz 1 1",
                ]
            )
            + "\n"
        )
        with open(pdir / "order.txt", "w", encoding="utf-8") as fh:
            for step, order in enumerate(orders):
                fh.write(f"{step} {order[0]} {order[1]}\n")

    data_file = tmp_path / "infretis_data.txt"
    data_file.write_text(
        "# header\n"
        "1 2 5.0 0.1 0.2 0.3 0.4\n"
        "2 2 4.0 0.5 0.6 0.7 0.8\n"
    )

    assert refresh_datafile_maxop(data_file, load_dir, DummyOrderFunction())

    lines = [ln for ln in data_file.read_text().splitlines() if not ln.startswith("#")]
    cols0 = lines[0].split()
    cols1 = lines[1].split()
    assert cols0[2] == "50.000000"
    assert cols1[2] == "40.000000"
