#!/usr/bin/env python3
"""
export_universe.py
==================
Exports a scipy sparse coverage matrix to the binary CSR format consumed by
main.cpp.

Binary layout (all little-endian):
    uint32_t  n_points          # number of rows  (prime points)
    uint32_t  n_lines           # number of columns (candidate lines)
    uint64_t  nnz               # total nonzeros
    uint64_t  indptr[n_points+1]  # row-pointer array
    uint32_t  indices[nnz]       # column indices (0-based line ids)

Usage
-----
    python export_universe.py \\
        --input  coverage_matrix.npz \\
        --output matrix.bin \\
        [--n_max 50000]            # truncate to first N rows (optional)
        [--verify]                 # round-trip verify after writing

The script accepts:
    * A .npz file containing a single sparse matrix stored with
      scipy.sparse.save_npz / scipy.sparse.load_npz.
    * A raw scipy CSR or CSC matrix passed programmatically (see
      ``export_matrix()`` function below).

Dependencies: numpy, scipy
"""

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp


# ─────────────────────────────────────────────────────────────────────────────
#  Core export function
# ─────────────────────────────────────────────────────────────────────────────

def export_matrix(
    mat: sp.spmatrix,
    output_path: str | Path,
    n_max: int | None = None,
    verbose: bool = True,
) -> None:
    """
    Write *mat* (or its first ``n_max`` rows) to a binary CSR file.

    Parameters
    ----------
    mat         : Any scipy sparse matrix.  Will be converted to CSR.
    output_path : Destination file path.
    n_max       : If given, only the first n_max rows are exported.
    verbose     : Print progress information.
    """
    output_path = Path(output_path)

    # ── Normalise to CSR ────────────────────────────────────────────────────
    if not sp.isspmatrix_csr(mat):
        if verbose:
            print(f"[INFO] Converting {type(mat).__name__} → CSR …", flush=True)
        mat = mat.tocsr()

    if n_max is not None and n_max < mat.shape[0]:
        if verbose:
            print(f"[INFO] Truncating to first {n_max} rows …", flush=True)
        mat = mat[:n_max]

    n_points = int(mat.shape[0])
    n_lines  = int(mat.shape[1])
    nnz      = int(mat.nnz)

    if n_points > 0xFFFF_FFFF or n_lines > 0xFFFF_FFFF:
        raise ValueError("Dimensions exceed uint32 range.")
    if nnz > 0xFFFF_FFFF_FFFF_FFFF:
        raise ValueError("nnz exceeds uint64 range.")

    # Ensure correct dtypes.
    indptr  = mat.indptr.astype(np.uint64)   # (n_points+1,)
    indices = mat.indices.astype(np.uint32)   # (nnz,)

    if verbose:
        print(
            f"[INFO] Exporting: {n_points:,} points × {n_lines:,} lines, "
            f"nnz={nnz:,}  →  {output_path}"
        )

    t0 = time.perf_counter()
    with open(output_path, "wb") as f:
        # Header: n_points (u32), n_lines (u32), nnz (u64)
        f.write(struct.pack("<IIQ", n_points, n_lines, nnz))

        # Row pointers
        f.write(indptr.tobytes())

        # Column indices
        f.write(indices.tobytes())

    elapsed = time.perf_counter() - t0
    size_mb  = output_path.stat().st_size / (1024 ** 2)

    if verbose:
        print(
            f"[INFO] Written in {elapsed:.2f}s  |  "
            f"file size = {size_mb:.1f} MB"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Round-trip verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_binary(bin_path: str | Path, mat: sp.csr_matrix,
                  n_max: int | None = None) -> bool:
    """
    Re-read the binary file and compare against the original matrix.
    Returns True on success.
    """
    bin_path = Path(bin_path)

    if n_max is not None and n_max < mat.shape[0]:
        mat = mat[:n_max]
    if not sp.isspmatrix_csr(mat):
        mat = mat.tocsr()

    with open(bin_path, "rb") as f:
        header = f.read(4 + 4 + 8)
        n_pts_r, n_lines_r, nnz_r = struct.unpack("<IIQ", header)

        indptr_r  = np.frombuffer(f.read((n_pts_r + 1) * 8), dtype=np.uint64)
        indices_r = np.frombuffer(f.read(nnz_r * 4),         dtype=np.uint32)

    ok = True
    if n_pts_r != mat.shape[0]:
        print(f"[VERIFY FAIL] n_points: {n_pts_r} != {mat.shape[0]}")
        ok = False
    if n_lines_r != mat.shape[1]:
        print(f"[VERIFY FAIL] n_lines: {n_lines_r} != {mat.shape[1]}")
        ok = False
    if nnz_r != mat.nnz:
        print(f"[VERIFY FAIL] nnz: {nnz_r} != {mat.nnz}")
        ok = False
    if not np.array_equal(indptr_r, mat.indptr.astype(np.uint64)):
        print("[VERIFY FAIL] indptr mismatch")
        ok = False
    if not np.array_equal(indices_r, mat.indices.astype(np.uint32)):
        print("[VERIFY FAIL] indices mismatch")
        ok = False

    if ok:
        print("[VERIFY OK] Binary file matches original matrix.")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
#  Example: building a coverage matrix from scratch (for testing)
# ─────────────────────────────────────────────────────────────────────────────

def build_toy_matrix(n_points: int = 20, n_lines: int = 50,
                     density: float = 0.15) -> sp.csr_matrix:
    """
    Construct a random binary coverage matrix for smoke-testing.
    In production, replace this with your actual structural-universe builder.
    """
    rng  = np.random.default_rng(seed=42)
    data = rng.random((n_points, n_lines)) < density

    # Guarantee every point is covered by at least one line.
    for i in range(n_points):
        if not data[i].any():
            data[i, rng.integers(n_lines)] = True

    return sp.csr_matrix(data.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a scipy sparse coverage matrix to binary CSR format."
    )
    parser.add_argument(
        "--input", "-i",
        help="Path to .npz file (scipy sparse). "
             "If omitted, a toy test matrix is generated.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output binary file path (e.g. matrix.bin).",
    )
    parser.add_argument(
        "--n_max", type=int, default=None,
        help="Export only the first N_MAX rows (points).",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Round-trip verify the output file after writing.",
    )
    args = parser.parse_args()

    # ── Load or generate matrix ─────────────────────────────────────────────
    if args.input:
        p = Path(args.input)
        if not p.exists():
            print(f"[FATAL] Input file not found: {p}", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] Loading sparse matrix from {p} …", flush=True)
        mat = sp.load_npz(str(p))
        print(f"[INFO] Loaded shape: {mat.shape}, nnz={mat.nnz:,}")
    else:
        print("[INFO] No --input provided. Generating toy matrix …")
        mat = build_toy_matrix()
        print(f"[INFO] Toy matrix shape: {mat.shape}, nnz={mat.nnz}")

    # ── Export ──────────────────────────────────────────────────────────────
    export_matrix(mat, args.output, n_max=args.n_max)

    # ── Verify ──────────────────────────────────────────────────────────────
    if args.verify:
        ok = verify_binary(args.output, mat, n_max=args.n_max)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
