"""The isometry block (villa #2279) on synthetic grids: exact, stretched, with a hole, through the writer and
the command line. No downloads."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import tifffile

from vesuvius.tifxyz import Tifxyz, read_tifxyz, write_tifxyz
from vesuvius.tifxyz.isometry import (
    ISOMETRY_VERSION,
    compute_grid_isometry,
    isometry_for_surface,
    main,
    read_segment_grid,
    store_isometry_meta,
    valid_nodes,
    write_isometry_meta,
)

STEP = 20.0
H, W = 40, 30
COLS, ROWS = H * (W - 1), (H - 1) * W  # edges along columns and along rows of a full grid


def plane(h=H, w=W):
    r, c = np.mgrid[0:h, 0:w].astype(float)
    return np.stack([1000 + c * STEP, 2000 + r * STEP, np.full_like(r, 500.0)], -1)


def block(xyz, **kw):
    return compute_grid_isometry(xyz, (STEP, STEP), **kw)


def test_exact_grid_is_isometric():
    r = block(plane())
    assert r["version"] == ISOMETRY_VERSION
    assert r["step_vx"] == [STEP, STEP]
    assert r["edges"] == COLS + ROWS
    assert (r["within_10pct"], r["within_25pct"], r["median_abs_log_ratio"]) == (1.0, 1.0, 0.0)
    assert (r["area_ratio"], r["angle_deg"], r["stretched_nodes"]) == (1.0, [90.0, 90.0, 90.0], 0)
    assert r["stretched_area_vx2"] == 0.0
    assert (r["l2_mean"], r["l2_median"], r["linf"], r["area_error"]) == (1.0, 1.0, 1.0, 0.0)
    assert "max_stretch" not in r


def test_uniform_stretch_along_columns():
    # 1.3x along the columns: those edges fail 10 % and 25 % but stay under the 1.5x flag
    xyz = plane()
    xyz[..., 0] = 1000 + (xyz[..., 0] - 1000) * 1.3
    r = block(xyz)
    assert r["edges"] == COLS + ROWS
    assert r["within_10pct"] == round(ROWS / (COLS + ROWS), 4)
    assert r["within_25pct"] == round(ROWS / (COLS + ROWS), 4)
    assert r["stretched_nodes"] == 0
    assert r["area_ratio"] == pytest.approx(1.3, abs=1e-4)
    assert r["linf"] == pytest.approx(1.3, abs=1e-4)
    assert r["l2_mean"] == pytest.approx(math.sqrt((1.3**2 + 1) / 2), abs=1e-4)
    assert r["angle_deg"] == [90.0, 90.0, 90.0]
    # uniform stretch: every triangle keeps its share of the area
    assert r["area_error"] == 0.0


def test_invalid_node_drops_its_edges_and_cells_without_counting_as_stretch():
    xyz = plane()
    xyz[20, 15] = -1.0
    assert not valid_nodes(xyz)[20, 15]
    r = block(xyz)
    assert r["edges"] == COLS + ROWS - 4
    assert (r["within_10pct"], r["stretched_nodes"], r["l2_mean"]) == (1.0, 0, 1.0)
    # the same through an explicit mask, with the point itself left in place
    valid = np.ones((H, W), dtype=bool)
    valid[20, 15] = False
    assert block(plane(), valid=valid) == r


def test_planted_jump_flags_both_ends_and_squash_counts_like_stretch():
    xyz = plane()
    xyz[:, 15:, 0] += 2 * STEP
    r = block(xyz)
    assert r["stretched_nodes"] == 2 * H
    assert r["stretched_area_vx2"] == 2 * H * STEP * STEP
    assert r["within_10pct"] == round((COLS + ROWS - H) / (COLS + ROWS), 4)
    xyz = plane()
    xyz[:, 15:, 0] -= 0.5 * STEP
    assert block(xyz)["stretched_nodes"] == 2 * H


def test_other_max_stretch_is_recorded():
    xyz = plane()
    xyz[:, 15:, 0] += 0.3 * STEP  # 1.3x: under 1.5, over 1.2
    assert block(xyz)["stretched_nodes"] == 0
    r = block(xyz, max_stretch=1.2)
    assert r["stretched_nodes"] == 2 * H
    assert r["max_stretch"] == 1.2


def test_degenerate_grids():
    r = block(plane(1, 5))  # one row: edges only
    assert r["edges"] == 4
    assert r["within_10pct"] == 1.0
    assert r["area_ratio"] is None and r["l2_mean"] is None
    r = block(plane(1, 1))
    assert r["edges"] == 0 and r["within_10pct"] is None
    r = block(np.full((3, 3, 3), -1.0))  # all holes
    assert r["edges"] == 0 and r["stretched_nodes"] == 0 and r["area_ratio"] is None


def test_bad_inputs():
    with pytest.raises(ValueError):
        compute_grid_isometry(plane()[..., :2], (STEP, STEP))
    with pytest.raises(ValueError):
        compute_grid_isometry(plane(), (0.0, STEP))
    with pytest.raises(ValueError):
        block(plane(), max_stretch=1.0)
    with pytest.raises(ValueError):
        block(plane(), valid=np.ones((H, W - 1), dtype=bool))


def test_store_isometry_meta_keeps_other_keys():
    meta = {"uuid": "seg", "scale": [1 / STEP, 1 / STEP]}
    out = store_isometry_meta(meta, block(plane()))
    assert out is meta
    assert meta["uuid"] == "seg" and meta["isometry"]["version"] == ISOMETRY_VERSION


def _surface(xyz=None):
    xyz = plane() if xyz is None else xyz
    return Tifxyz(
        _x=xyz[..., 0].astype(np.float32),
        _y=xyz[..., 1].astype(np.float32),
        _z=xyz[..., 2].astype(np.float32),
        uuid="synthetic",
        _scale=(1 / STEP, 1 / STEP),
    )


def test_surface_block_matches_the_grid_block():
    assert isometry_for_surface(_surface()) == block(plane())


def test_writer_stores_the_block_and_reader_returns_it(tmp_path: Path):
    out = write_tifxyz(tmp_path / "seg", _surface())
    meta = json.loads((out / "meta.json").read_text())
    assert meta["isometry"]["version"] == ISOMETRY_VERSION
    assert meta["isometry"]["within_10pct"] == 1.0
    assert meta["scale"] == [1 / STEP, 1 / STEP]
    surface = read_tifxyz(out)
    assert surface.extra["isometry"] == meta["isometry"]
    # re-saving a read surface refreshes the block instead of carrying the old one in extra
    surface._x[:, 15:] += 2 * STEP
    out2 = write_tifxyz(tmp_path / "seg2", surface)
    meta2 = json.loads((out2 / "meta.json").read_text())
    assert meta2["isometry"]["stretched_nodes"] == 2 * H


def test_writer_can_skip_the_block(tmp_path: Path):
    out = write_tifxyz(tmp_path / "seg", _surface(), write_isometry=False)
    assert "isometry" not in json.loads((out / "meta.json").read_text())


def _write_folder(folder: Path, xyz, meta=None, indent=4):
    folder.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate("xyz"):
        tifffile.imwrite(str(folder / f"{c}.tif"), xyz[..., i].astype(np.float32))
    meta = {"uuid": "out", "scale": [1 / STEP, 1 / STEP], "format": "tifxyz", "type": "seg"} if meta is None else meta
    (folder / "meta.json").write_text(json.dumps(meta, indent=indent))
    return folder


def test_read_segment_grid_uses_mask_when_present(tmp_path: Path):
    folder = _write_folder(tmp_path / "seg", plane())
    xyz, valid, meta = read_segment_grid(folder)
    assert xyz.shape == (H, W, 3) and valid.all() and meta["uuid"] == "out"
    mask = np.full((H, W), 255, dtype=np.uint8)
    mask[20, 15] = 0
    tifffile.imwrite(str(folder / "mask.tif"), mask)
    _, valid, _ = read_segment_grid(folder)
    assert not valid[20, 15] and valid.sum() == H * W - 1


def test_cli_prints_and_writes_keeping_other_keys(tmp_path: Path, capsys):
    folder = _write_folder(tmp_path / "seg", plane())
    assert main([str(folder)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["within_10pct"] == 1.0
    assert "isometry" not in json.loads((folder / "meta.json").read_text())

    assert main([str(folder), "--write"]) == 0
    text = (folder / "meta.json").read_text()
    meta = json.loads(text)
    assert meta["isometry"] == printed
    assert (meta["uuid"], meta["format"], meta["type"]) == ("out", "tifxyz", "seg")
    assert text.startswith("{\n    ")  # the file's own indent is kept
    assert not [p for p in folder.iterdir() if p.name.startswith(".meta_")]  # no temp file left behind

    # a second write replaces the block
    xyz = plane()
    xyz[:, 15:, 0] += 2 * STEP
    for i, c in enumerate("xyz"):
        tifffile.imwrite(str(folder / f"{c}.tif"), xyz[..., i].astype(np.float32))
    assert main([str(folder), "--write"]) == 0
    assert json.loads((folder / "meta.json").read_text())["isometry"]["stretched_nodes"] == 2 * H


def test_cli_exit_codes(tmp_path: Path, capsys):
    with pytest.raises(SystemExit) as e:
        main([str(tmp_path / "missing")])
    assert e.value.code == 2
    folder = _write_folder(tmp_path / "noscale", plane(), meta={"uuid": "x"})
    with pytest.raises(SystemExit) as e:
        main([str(folder)])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        main([str(folder), "--max-stretch", "1"])
    assert e.value.code == 2
    # a grid with no complete cell prints the edge statistics and exits 1
    folder = _write_folder(tmp_path / "line", plane(1, 5))
    assert main([str(folder)]) == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["edges"] == 4 and printed["area_ratio"] is None


def test_write_isometry_meta_without_existing_meta(tmp_path: Path):
    folder = tmp_path / "seg"
    folder.mkdir()
    write_isometry_meta(folder, block(plane()))
    assert json.loads((folder / "meta.json").read_text())["isometry"]["edges"] == COLS + ROWS
