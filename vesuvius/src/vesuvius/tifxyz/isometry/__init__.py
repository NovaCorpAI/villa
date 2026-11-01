"""Grid isometry of a tifxyz segment: how well the saved grid keeps the step its ``scale`` promises.

A tifxyz ``meta.json`` carries ``scale``, nodes per voxel along the grid's columns and rows, which promises a
step of ``1 / scale`` voxels between neighbouring nodes. Nothing in the file records whether the grid keeps that
promise. This module computes the ``isometry`` block proposed in ScrollPrize/villa#2279 (version 1): every grid
edge against the promised step, triangle area against the promised cell area, the grid angle, the nodes touching
an edge more than ``max_stretch`` times too long or too short, and flatboi's stretch metrics with the grid itself
as the 2D coordinates (column x step, row x step), so a perfect grid gives L2 = Linf = 1.

The numbers match ``kit tifxyz isometry`` from https://github.com/ChaseHendrick/Scrolls, which first computed
this block, and the thresholds are fixed per version so segments stay comparable.

Command line::

    python -m vesuvius.tifxyz.isometry <segment.tifxyz>          # print the block
    python -m vesuvius.tifxyz.isometry <segment.tifxyz> --write  # also store it in meta.json

Only numpy and tifffile are needed; the grid is never upsampled or modified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import tifffile

if TYPE_CHECKING:  # pragma: no cover
    from .types import Tifxyz

ISOMETRY_VERSION = 1
DEFAULT_MAX_STRETCH = 1.5

__all__ = [
    "ISOMETRY_VERSION",
    "DEFAULT_MAX_STRETCH",
    "compute_grid_isometry",
    "isometry_for_surface",
    "store_isometry_meta",
    "read_segment_grid",
    "write_isometry_meta",
    "valid_nodes",
    "main",
]


# ---------------------------------------------------------------------------- validity


def valid_nodes(xyz: np.ndarray) -> np.ndarray:
    """Nodes that hold a point: finite, not the ``(-1, -1, -1)`` sentinel and not all zero.

    This is the rule ``SurfaceArea.hpp`` applies cell by cell (non-finite or sentinel corners are skipped),
    plus the all-zero node some writers leave in place of a hole.
    """
    xyz = np.asarray(xyz)
    finite = np.isfinite(xyz).all(-1)
    sentinel = (xyz == -1.0).all(-1)
    zero = (xyz == 0.0).all(-1)
    return finite & ~sentinel & ~zero


# ---------------------------------------------------------------------------- geometry


def _edge_ratios(P: np.ndarray, valid: np.ndarray, step: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """(along columns (H, W-1), along rows (H-1, W)): edge length over the promised step, NaN at a hole."""
    su, sv = step
    with np.errstate(invalid="ignore"):
        ec = np.linalg.norm(np.diff(P, axis=1), axis=-1) / su
        er = np.linalg.norm(np.diff(P, axis=0), axis=-1) / sv
    ec[~(valid[:, :-1] & valid[:, 1:])] = np.nan
    er[~(valid[:-1, :] & valid[1:, :])] = np.nan
    return ec, er


def _cells(P: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(complete cells (H-1, W-1), area of each cell in voxels^2 from two triangles)."""
    c = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    p00, p01, p10, p11 = P[:-1, :-1], P[:-1, 1:], P[1:, :-1], P[1:, 1:]
    with np.errstate(invalid="ignore"):
        t1 = np.cross(p01 - p00, p11 - p00)
        t2 = np.cross(p11 - p00, p10 - p00)
        area = 0.5 * (np.linalg.norm(t1, axis=-1) + np.linalg.norm(t2, axis=-1))
        nn = np.linalg.norm(t1 + t2, axis=-1)
    area = np.where(c, area, 0.0)
    c = c & (nn > 0)
    return c, area


def _tri_stretch(
    q1: np.ndarray,
    q2: np.ndarray,
    q3: np.ndarray,
    uv1: Tuple[float, float],
    uv2: Tuple[float, float],
    uv3: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """flatboi's ``triangle_stretch`` over arrays of triangles: (L2, Linf, area 3D, area 2D).

    Sander et al.'s stretch of the map from the 2D coordinates to the 3D points; 1 and 1 for an isometric
    triangle. Same formulas as ``volume-cartographer/libs/flatboi/flatboi.cpp``.
    """
    (s1, t1), (s2, t2), (s3, t3) = uv1, uv2, uv3
    A = ((s2 - s1) * (t3 - t1) - (s3 - s1) * (t2 - t1)) / 2.0
    Ss = (q1 * (t2 - t3) + q2 * (t3 - t1) + q3 * (t1 - t2)) / (2.0 * A)
    St = (q1 * (s3 - s2) + q2 * (s1 - s3) + q3 * (s2 - s1)) / (2.0 * A)
    a = (Ss * Ss).sum(-1)
    b = (Ss * St).sum(-1)
    c = (St * St).sum(-1)
    G = np.sqrt(((a + c) + np.sqrt((a - c) ** 2 + 4.0 * b * b)) / 2.0)
    L2 = np.sqrt((a + c) / 2.0)
    ab = np.linalg.norm(q2 - q1, axis=-1)
    bc = np.linalg.norm(q3 - q2, axis=-1)
    ca = np.linalg.norm(q1 - q3, axis=-1)
    s = (ab + bc + ca) / 2.0
    a3 = np.sqrt(np.maximum(0.0, s * (s - ab) * (s - bc) * (s - ca)))
    return L2, G, a3, np.full_like(a3, abs(A))


def _weighted_median(data: np.ndarray, weights: np.ndarray) -> float:
    """flatboi's ``weighted_median``: the first value whose cumulative weight reaches half the total."""
    o = np.argsort(data, kind="stable")
    acc = np.cumsum(weights[o])
    return float(data[o][min(int(np.searchsorted(acc, 0.5 * acc[-1])), len(o) - 1)])


def _stretched_nodes(ec: np.ndarray, er: np.ndarray, valid: np.ndarray, max_stretch: float) -> np.ndarray:
    """Nodes touching an edge more than ``max_stretch`` times longer or shorter than the step."""
    H, W = valid.shape
    with np.errstate(invalid="ignore"):
        bc = (ec > max_stretch) | (ec < 1.0 / max_stretch)
        br = (er > max_stretch) | (er < 1.0 / max_stretch)
    bad = np.zeros((H, W), dtype=bool)
    bad[:, :-1] |= bc
    bad[:, 1:] |= bc
    bad[:-1, :] |= br
    bad[1:, :] |= br
    return bad & valid


# ---------------------------------------------------------------------------- the block


def compute_grid_isometry(
    xyz: np.ndarray,
    step: Sequence[float],
    valid: Optional[np.ndarray] = None,
    *,
    max_stretch: float = DEFAULT_MAX_STRETCH,
) -> Dict[str, Any]:
    """The ``isometry`` block (version 1) for one grid of 3D points.

    Parameters
    ----------
    xyz : np.ndarray
        Grid points, shape ``(H, W, 3)``, in voxels of one volume. Any float dtype; computed in float64.
    step : Sequence[float]
        Promised step in voxels between neighbouring nodes, ``(step along columns, step along rows)``, which
        is ``(1 / scale[0], 1 / scale[1])`` for the ``scale`` stored in ``meta.json``.
    valid : Optional[np.ndarray]
        Boolean ``(H, W)`` mask of nodes that hold a point. Default: :func:`valid_nodes`.
    max_stretch : float
        A node is "stretched" when one of its edges is more than this many times longer or shorter than the
        step. Fixed at 1.5 for version 1; passing another value is allowed but the block then says so.

    Returns
    -------
    Dict[str, Any]
        Keys ``version``, ``step_vx``, ``edges``, ``within_10pct``, ``within_25pct``,
        ``median_abs_log_ratio``, ``area_ratio``, ``angle_deg``, ``stretched_nodes``,
        ``stretched_area_vx2``, ``l2_mean``, ``l2_median``, ``linf``, ``area_error``. Statistics that need
        at least one edge or one complete cell are ``None`` when the grid has none. Values are rounded to
        four decimals (angles to one) so the stored block is stable across machines.
    """
    P = np.asarray(xyz, dtype=np.float64)
    if P.ndim != 3 or P.shape[-1] != 3:
        raise ValueError(f"xyz must have shape (H, W, 3), got {P.shape}")
    su, sv = (float(step[0]), float(step[1]))
    if not (math.isfinite(su) and math.isfinite(sv) and su > 0 and sv > 0):
        raise ValueError(f"step must be two finite numbers above 0, got {tuple(step)!r}")
    if not max_stretch > 1:
        raise ValueError(f"max_stretch must be above 1, got {max_stretch}")
    if valid is None:
        valid = valid_nodes(P)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != P.shape[:2]:
        raise ValueError(f"valid must have shape {P.shape[:2]}, got {valid.shape}")

    out: Dict[str, Any] = {
        "version": ISOMETRY_VERSION,
        "step_vx": [round(su, 4), round(sv, 4)],
        "edges": 0,
        "within_10pct": None,
        "within_25pct": None,
        "median_abs_log_ratio": None,
        "area_ratio": None,
        "angle_deg": None,
        "stretched_nodes": 0,
        "stretched_area_vx2": 0.0,
        "l2_mean": None,
        "l2_median": None,
        "linf": None,
        "area_error": None,
    }
    if max_stretch != DEFAULT_MAX_STRETCH:
        out["max_stretch"] = max_stretch

    H, W = valid.shape
    if H < 2 and W < 2:
        return out

    ec, er = _edge_ratios(P, valid, (su, sv))
    both = np.concatenate([ec[np.isfinite(ec)], er[np.isfinite(er)]])
    out["edges"] = int(both.size)
    if both.size:
        out["within_10pct"] = round(float(((both > 1 / 1.1) & (both < 1.1)).mean()), 4)
        out["within_25pct"] = round(float(((both > 1 / 1.25) & (both < 1.25)).mean()), 4)
        with np.errstate(divide="ignore"):
            out["median_abs_log_ratio"] = round(float(np.median(np.abs(np.log(both)))), 4)

    k = int(_stretched_nodes(ec, er, valid, max_stretch).sum())
    out["stretched_nodes"] = k
    out["stretched_area_vx2"] = round(k * su * sv, 1)

    if H < 2 or W < 2:
        return out

    c, area = _cells(P, valid)
    if not c.any():
        return out

    with np.errstate(invalid="ignore"):
        u = P[:-1, 1:] - P[:-1, :-1]
        w = P[1:, :-1] - P[:-1, :-1]
        cos = (u * w).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(w, axis=-1))
        ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))[c]
    out["area_ratio"] = round(float(area[c].sum() / (c.sum() * su * sv)), 4)
    out["angle_deg"] = [round(float(v), 1) for v in np.percentile(ang, (5, 50, 95))]

    p00, p01, p10, p11 = P[:-1, :-1][c], P[:-1, 1:][c], P[1:, :-1][c], P[1:, 1:][c]
    o, x, y, xy = (0.0, 0.0), (su, 0.0), (0.0, sv), (su, sv)
    r1 = _tri_stretch(p00, p01, p11, o, x, xy)
    r2 = _tri_stretch(p00, p11, p10, o, xy, y)
    L2, G, a3, a2 = (np.concatenate([a, b]) for a, b in zip(r1, r2))
    l2sq = L2 * L2
    a3_sum = float(a3.sum())
    if a3_sum > 0:
        out["l2_mean"] = round(float(np.sqrt((l2sq * a3).sum() / a3_sum)), 4)
        out["l2_median"] = round(_weighted_median(l2sq, a3), 4)
        out["linf"] = round(float(G.max()), 4)
        al, be = a3 / a3_sum, a2 / a2.sum()
        err = np.where(al > be, 1 - be / (al + 1e-30), 1 - al / (be + 1e-30))
        out["area_error"] = round(float(err.mean()), 4)
    return out


def store_isometry_meta(meta: Dict[str, Any], block: Dict[str, Any]) -> Dict[str, Any]:
    """Put ``block`` under ``meta["isometry"]`` and return ``meta`` (the C++ ``storeIsometryMeta`` shape)."""
    meta["isometry"] = dict(block)
    return meta


def isometry_for_surface(surface: "Tifxyz", *, max_stretch: float = DEFAULT_MAX_STRETCH) -> Dict[str, Any]:
    """The block for an in-memory :class:`~vesuvius.tifxyz.Tifxyz`, on its stored grid.

    ``Tifxyz._scale`` is ``(scale_y, scale_x)``, so the promised step is ``(1 / scale_x, 1 / scale_y)``.
    """
    scale_y, scale_x = surface._scale
    xyz = np.stack([surface._x, surface._y, surface._z], axis=-1)
    return compute_grid_isometry(
        xyz, (1.0 / scale_x, 1.0 / scale_y), surface._valid_mask, max_stretch=max_stretch
    )


# ---------------------------------------------------------------------------- files


def _step_from_meta(meta: Dict[str, Any]) -> Tuple[float, float]:
    scale = meta.get("scale")
    try:
        sx, sy = (float(v) for v in scale[:2])
    except (TypeError, ValueError):
        raise ValueError(f"meta.json scale must be two numbers above 0, got {scale!r}") from None
    if not (math.isfinite(sx) and math.isfinite(sy) and sx > 0 and sy > 0):
        raise ValueError(f"meta.json scale must be two numbers above 0, got {scale!r}")
    return 1.0 / sx, 1.0 / sy


def read_segment_grid(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Read a tifxyz folder as ``(xyz (H, W, 3) float64, valid (H, W) bool, meta dict)``.

    ``mask.tif`` decides validity when present (non-zero is valid); otherwise :func:`valid_nodes` does.
    Reads the stored grid as-is, without the package's upsampling or label discovery, so it works with
    numpy and tifffile alone.
    """
    folder = Path(path)
    if not folder.is_dir():
        raise FileNotFoundError(f"no tifxyz folder at {folder}")
    arrs = []
    for c in "xyz":
        tif = folder / f"{c}.tif"
        if not tif.is_file():
            raise FileNotFoundError(f"missing {tif}")
        arrs.append(np.asarray(tifffile.imread(str(tif))))
    if any(a.ndim != 2 or a.shape != arrs[0].shape for a in arrs):
        raise ValueError(f"{folder}: x.tif, y.tif and z.tif must be 2D grids of one shape")
    xyz = np.stack(arrs, axis=-1).astype(np.float64)
    valid = valid_nodes(xyz)
    mask_path = folder / "mask.tif"
    if mask_path.is_file():
        mask = np.asarray(tifffile.imread(str(mask_path)))
        if mask.shape == valid.shape:
            valid &= mask != 0
    meta_path = folder / "meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.is_file():
        with open(meta_path, "r") as f:
            meta = json.load(f)
    return xyz, valid, meta


def _json_indent(text: str) -> int:
    """Indent of an existing meta.json: 4 (VC3D) or 2 (this package), judged from its first nested line."""
    for line in text.splitlines()[1:]:
        stripped = line.lstrip(" ")
        if stripped and stripped != line:
            return len(line) - len(stripped)
    return 4


def write_isometry_meta(path: Union[str, Path], block: Dict[str, Any]) -> Path:
    """Store ``block`` in the segment's ``meta.json``, keeping every other key, by an atomic replace."""
    folder = Path(path)
    meta_path = folder / "meta.json"
    text = meta_path.read_text() if meta_path.is_file() else "{}"
    meta = json.loads(text) if text.strip() else {}
    store_isometry_meta(meta, block)
    fd, tmp = tempfile.mkstemp(prefix=".meta_", suffix=".json", dir=str(folder))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(meta, f, indent=_json_indent(text))
            f.write("\n")
        os.replace(tmp, meta_path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return meta_path


# ---------------------------------------------------------------------------- command line


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m vesuvius.tifxyz.isometry",
        description="Compute the isometry block (villa #2279, version 1) of a tifxyz segment and print it as "
        "JSON. With --write, also store it under \"isometry\" in the segment's meta.json, leaving the grid and "
        "every other key untouched.",
    )
    p.add_argument("segment", help="path to a tifxyz folder (x.tif, y.tif, z.tif, meta.json)")
    p.add_argument("--write", action="store_true", help="store the block in meta.json")
    p.add_argument(
        "--max-stretch",
        type=float,
        default=DEFAULT_MAX_STRETCH,
        help="a stretched node has an edge this many times too long or too short (default %(default)s, "
        "the version 1 threshold; another value is recorded in the block)",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Exit 0 on success, 1 when the grid has no complete cell, 2 on bad input."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.max_stretch > 1:
        parser.error(f"--max-stretch must be above 1 (got {args.max_stretch})")
    try:
        xyz, valid, meta = read_segment_grid(args.segment)
        step = _step_from_meta(meta)
    except (FileNotFoundError, ValueError, OSError) as exc:
        parser.error(str(exc))
    block = compute_grid_isometry(xyz, step, valid, max_stretch=args.max_stretch)
    if args.write:
        write_isometry_meta(args.segment, block)
    json.dump(block, sys.stdout, indent=2)
    sys.stdout.write("\n")
    if block["area_ratio"] is None:
        print(f"{args.segment}: no complete grid cell, only edge statistics are defined", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
