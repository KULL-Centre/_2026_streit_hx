#!/usr/bin/env python3
"""
Convert HDX-MS cluster-data (CD) CSV files to peptide fractional uptake curves.

This version computes averages/errors from replicate-averaged absolute uptakes,
then converts to fractional uptake and propagates maxD uncertainty.

Core workflow
-------------
raw charge-state centroid rows
    -> convert each row to neutral mass
    -> optionally filter State by substring
    -> define independent measurements from exact File entries
    -> within each peptide / timepoint / measurement:
           average charge states and duplicate measurements
    -> for each peptide:
           calculate pooled undeuterated/time-zero mass
    -> for each deuterated measurement:
           uptake_Da = mass - pooled_t0_mass
    -> across independent measurements at each peptide/timepoint:
           mean absolute uptake ± SD
    -> estimate maxD from final timepoint or fitted mean uptake curve
    -> fractional uptake = uptake / maxD
    -> propagate uncertainty from both uptake and maxD

Input assumptions
-----------------
Required columns:
    Start, End, Sequence, Exposure, z, Center

Typical optional columns:
    Protein, State, File, Modification, Fragment, MaxUptake, MHP, RT, Inten

Output columns:
    Start,End,Sequence,Time (s),Rel. Uptake,Rel. Uptake SD

Replicate / measurement detection
---------------------------------
Independent measurements are ALWAYS defined by exact entries in the File column.
Rows belong to the same replicate only when their File values are identical.

If --state is provided:
    rows are first filtered by State substring, case-insensitive; among the
    retained rows, replicate identity is still defined only by exact File values.

The State column is never used to define replicate identity. It is only used for
optional filtering via --state.

Important
---------
This script does NOT require each File-defined replicate to contain its own
undeuterated/time-zero row. Time-zero rows are pooled per peptide, because
undeuterated controls may have separate names/files and HDX error is usually
dominated by the deuterated measurements.

Charge states are averaged within each File-defined independent measurement before
replicate statistics are calculated. This avoids treating different charge
states from the same file as independent replicates.

Fractional uptake uncertainty is propagated as a ratio using the derivative
form:

    rel = uptake / maxD

    var(rel) = (sd_uptake / maxD)^2
             + (uptake * sd_maxD / maxD^2)^2

This is equivalent to:

    rel * sqrt((sd_uptake / uptake)^2 + (sd_maxD / maxD)^2)

away from uptake == 0, but is numerically safe near uptake == 0.

Examples
--------
python convert_cd_hdx_uptake.py CD_file.csv -o uptake_curves.csv

python convert_cd_hdx_uptake.py CD_file.csv -o uptake_curves_ANP.csv --state ANP

python convert_cd_hdx_uptake.py CD_file.csv -o uptake_curves_fit.csv --maxd-method fit

python convert_cd_hdx_uptake.py CD_file.csv -o uptake_curves.csv --write-absolute-csv absolute_uptakes.csv

python convert_cd_hdx_uptake.py CD_file.csv -o uptake_curves_pooled.csv --pooled-relative-sd
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

PROTON_MASS = 1.007276466621  # Da


def stretched_exp(t: np.ndarray, plateau: float, k: float, beta: float) -> np.ndarray:
    """Stretched exponential uptake model."""
    return plateau * (1.0 - np.exp(-np.power(k * t, beta)))


def exposure_to_seconds(exposure: pd.Series, exposure_unit: str) -> pd.Series:
    """Convert input exposure times to seconds."""
    exposure = pd.to_numeric(exposure, errors="coerce")
    unit = exposure_unit.lower()

    if unit in {"s", "sec", "second", "seconds"}:
        return exposure
    if unit in {"ms", "millisecond", "milliseconds"}:
        return exposure / 1000.0
    if unit in {"min", "minute", "minutes"}:
        return exposure * 60.0

    raise ValueError(f"Unsupported exposure unit: {exposure_unit}")


def finite_values(x: pd.Series | np.ndarray) -> np.ndarray:
    """Return finite numeric values from a Series/array."""
    arr = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def finite_mean(x: pd.Series) -> float:
    """Mean using only finite values."""
    arr = finite_values(x)
    if len(arr) == 0:
        return np.nan
    return float(np.mean(arr))


def finite_std(x: pd.Series) -> float:
    """Sample SD using only finite values."""
    arr = finite_values(x)
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def n_finite(x: pd.Series) -> int:
    """Number of finite values."""
    return int(len(finite_values(x)))


def unique_nonempty_count(s: pd.Series) -> int:
    """Count non-empty unique string values."""
    vals = (
        s.astype(str)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
        .dropna()
        .unique()
    )
    return len(vals)


def fit_plateau(time_s: np.ndarray, uptake_da: np.ndarray) -> float:
    """
    Fit plateau for a mean uptake curve.

    Falls back to final finite uptake if the fit fails.
    """
    time_s = np.asarray(time_s, dtype=float)
    uptake_da = np.asarray(uptake_da, dtype=float)

    finite = np.isfinite(time_s) & np.isfinite(uptake_da) & (time_s > 0)
    t = time_s[finite]
    y = uptake_da[finite]

    if len(t) == 0:
        return np.nan

    if len(t) < 4 or np.nanmax(y) <= 0:
        return float(np.nanmax(y))

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    final_uptake = max(float(y[-1]), float(np.nanmax(y)), 1e-12)

    p0 = [
        final_uptake,
        1.0 / max(float(np.median(t)), 1e-12),
        0.7,
    ]
    bounds = ([0.0, 1e-12, 0.05], [np.inf, np.inf, 3.0])

    try:
        popt, _ = curve_fit(
            stretched_exp,
            t,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=20000,
        )
        plateau = float(popt[0])

        if not np.isfinite(plateau) or plateau <= 0:
            return final_uptake

        return plateau

    except Exception:
        return final_uptake


def add_measurement_replicate_column(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """
    Add a Replicate column for independent measurements within each timepoint.

    Replicate identity is ALWAYS the exact value in the File column.
    Rows belong to the same replicate only if their File entries are identical.

    State is not used to define replicates. If --state was provided, State is
    used only for filtering before this function is called.

    This deliberately does NOT require the same File value to exist across all
    timepoints. Timepoint-specific filenames are allowed; each distinct File
    value is treated as an independent measurement at its peptide/timepoint.
    """
    df = df.copy()

    if "File" not in df.columns:
        raise ValueError(
            "Input has no 'File' column. Replicate identity is defined only "
            "from exact File entries, so this column is required."
        )

    file_values = df["File"].astype(str).replace(
        {"nan": np.nan, "None": np.nan, "": np.nan}
    )

    if file_values.dropna().empty:
        raise ValueError(
            "The 'File' column contains no non-empty values. Replicate identity "
            "requires exact File entries."
        )

    df = df.loc[file_values.notna()].copy()
    df["Replicate"] = file_values.loc[df.index].astype(str)

    return df, "File"


def safe_ratio_sd(
    value_num: np.ndarray,
    sd_num: np.ndarray,
    value_den: float,
    sd_den: float,
) -> np.ndarray:
    """
    Propagate SD for ratio:

        r = num / den

    using:

        var(r) = (sd_num / den)^2 + (num * sd_den / den^2)^2

    This avoids division by zero at num == 0.
    """
    value_num = np.asarray(value_num, dtype=float)
    sd_num = np.asarray(sd_num, dtype=float)

    if not np.isfinite(value_den) or value_den <= 0:
        return np.full_like(value_num, np.nan, dtype=float)

    if not np.isfinite(sd_den):
        sd_den = 0.0

    var = (sd_num / value_den) ** 2 + ((value_num * sd_den) / (value_den**2)) ** 2
    var = np.where(np.isfinite(var) & (var >= 0), var, np.nan)

    return np.sqrt(var)


def convert_cd_to_uptake(
    input_csv: str | Path,
    output_csv: str | Path,
    exposure_unit: str = "ms",
    maxd_method: str = "final",
    state: str | None = None,
    min_replicates: int = 1,
    keep_time_zero: bool = False,
    write_absolute_csv: str | Path | None = None,
    pooled_relative_sd: bool = False,
    include_protein_in_identity: bool = True,
    include_modification_in_identity: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Convert one CD CSV file to fractional uptake curves."""
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    df = pd.read_csv(input_csv)

    required = ["Start", "End", "Sequence", "Exposure", "z", "Center"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    state_was_provided = state is not None

    # ------------------------------------------------------------------
    # Optional State substring filtering.
    # ------------------------------------------------------------------
    if state is not None:
        if "State" not in df.columns:
            raise ValueError("--state was provided, but the input has no 'State' column")

        keep = df["State"].astype(str).str.contains(str(state), case=False, na=False)
        df = df.loc[keep].copy()

        if df.empty:
            raise ValueError(f"No rows remain after filtering State contains {state!r}")

    df = df.copy()

    # ------------------------------------------------------------------
    # Numeric conversion and neutral mass.
    # ------------------------------------------------------------------
    df["z"] = pd.to_numeric(df["z"], errors="coerce")
    df["Center"] = pd.to_numeric(df["Center"], errors="coerce")
    df["Time (s)"] = exposure_to_seconds(df["Exposure"], exposure_unit)

    # Convert centroid m/z to neutral mass.
    df["Mass"] = df["z"] * df["Center"] - df["z"] * PROTON_MASS

    # Drop rows that cannot contribute.
    before = len(df)
    df = df[np.isfinite(df["Mass"]) & np.isfinite(df["Time (s)"])].copy()
    after = len(df)

    if df.empty:
        raise ValueError("No finite Mass / Time rows remain after cleaning.")

    if verbose and after < before:
        print(f"Dropped {before - after} rows with non-finite Mass or Time.")

    # ------------------------------------------------------------------
    # Peptide identity.
    #
    # File is NOT included here because it defines measurements/replicates.
    # State is NOT included here because it may be used only as a filter.
    # z is NOT included because charge states must be averaged within
    # measurement/timepoint.
    # ------------------------------------------------------------------
    peptide_cols = ["Start", "End", "Sequence"]
    identity_cols = peptide_cols.copy()

    if include_protein_in_identity and "Protein" in df.columns:
        identity_cols.append("Protein")

    if include_modification_in_identity:
        for c in ["Modification", "PTM_ID"]:
            if c in df.columns:
                identity_cols.append(c)

    # ------------------------------------------------------------------
    # Independent measurement identity.
    # ------------------------------------------------------------------
    df, replicate_source = add_measurement_replicate_column(df)

    if verbose:
        print(f"Using independent measurement identity from: {replicate_source}")
        print(f"Peptide identity columns: {identity_cols}")
        print(f"Unique File entries across all rows: {df['Replicate'].nunique()}")

    # ------------------------------------------------------------------
    # 1. Average charge states / duplicate features within each
    #    peptide / timepoint / independent measurement.
    #
    # This prevents different charge states from being treated as independent
    # replicates.
    # ------------------------------------------------------------------
    within_rep = (
        df.groupby(identity_cols + ["Replicate", "Time (s)"], dropna=False)
        .agg(
            Mass_mean=("Mass", finite_mean),
            Mass_sd_within_rep=("Mass", finite_std),
            n_charge_state_rows=("Mass", n_finite),
        )
        .reset_index()
    )

    within_rep = within_rep[np.isfinite(within_rep["Mass_mean"])].copy()

    if within_rep.empty:
        raise ValueError("No measurement-level mass means could be calculated.")

    # ------------------------------------------------------------------
    # 2. Calculate pooled undeuterated / time-zero mass per peptide.
    #
    # This does NOT require every File/State measurement to have its own t0.
    # All t0 rows for the peptide are averaged after charge-state averaging.
    # ------------------------------------------------------------------
    t0 = within_rep[within_rep["Time (s)"] == 0].copy()

    if t0.empty:
        raise ValueError(
            "No time-zero rows found. Cannot calculate absolute uptake. "
            "Check Exposure values and --exposure-unit."
        )

    t0_mass = (
        t0.groupby(identity_cols, dropna=False)
        .agg(
            Mass0=("Mass_mean", finite_mean),
            Mass0_SD=("Mass_mean", finite_std),
            n_t0=("Mass_mean", n_finite),
        )
        .reset_index()
    )

    if verbose:
        print(f"Peptides with time-zero mass: {len(t0_mass)}")

    # ------------------------------------------------------------------
    # 3. Keep deuterated rows and subtract pooled t0 mass.
    # ------------------------------------------------------------------
    if keep_time_zero:
        deut = within_rep.copy()
    else:
        deut = within_rep[within_rep["Time (s)"] > 0].copy()

    if deut.empty:
        raise ValueError("No deuterated timepoints found after removing time zero.")

    deut = deut.merge(t0_mass, on=identity_cols, how="left")

    missing_t0 = deut["Mass0"].isna().sum()
    if missing_t0 and verbose:
        print(f"Dropping {missing_t0} measurement rows without matching peptide t0 mass.")

    deut = deut[np.isfinite(deut["Mass0"])].copy()

    if deut.empty:
        raise ValueError(
            "No deuterated rows could be matched to peptide-specific t0 masses."
        )

    deut["Uptake_Da"] = deut["Mass_mean"] - deut["Mass0"]

    # ------------------------------------------------------------------
    # 4. Average absolute uptake across independent measurements at each
    #    peptide/timepoint.
    # ------------------------------------------------------------------
    abs_mean = (
        deut.groupby(identity_cols + ["Time (s)"], dropna=False)
        .agg(
            Uptake_Da_mean=("Uptake_Da", finite_mean),
            Uptake_Da_SD=("Uptake_Da", finite_std),
            n_replicates=("Uptake_Da", n_finite),
            Mass0=("Mass0", finite_mean),
            Mass0_SD=("Mass0_SD", finite_mean),
            n_t0=("n_t0", finite_mean),
        )
        .reset_index()
    )

    if min_replicates > 1:
        before_minrep = len(abs_mean)
        abs_mean = abs_mean[abs_mean["n_replicates"] >= min_replicates].copy()
        after_minrep = len(abs_mean)

        if verbose and after_minrep < before_minrep:
            print(
                f"Dropped {before_minrep - after_minrep} peptide/timepoint rows "
                f"with fewer than {min_replicates} replicates."
            )

    if abs_mean.empty:
        raise ValueError(
            "No peptide/timepoint groups remain after applying --min-replicates."
        )

    # ------------------------------------------------------------------
    # 5. Estimate maxD and maxD_SD.
    #
    # For --maxd-method final:
    #     maxD is the mean uptake at the final timepoint.
    #     maxD_SD is the replicate SD at that same final timepoint.
    #
    # For --maxd-method fit:
    #     maxD is the fitted plateau of the mean uptake curve.
    #     maxD_SD is approximated by final-timepoint replicate SD.
    #
    # Since File values may not define longitudinal replicate curves across
    # all timepoints, this does not attempt replicate-wise curve fitting.
    # ------------------------------------------------------------------
    maxd_records = []

    for keys, g_abs in abs_mean.groupby(identity_cols, dropna=False, sort=False):
        g_abs = g_abs.sort_values("Time (s)").copy()

        if not isinstance(keys, tuple):
            keys = (keys,)

        record = {col: val for col, val in zip(identity_cols, keys)}

        g_nonzero = g_abs[g_abs["Time (s)"] > 0].copy()

        if g_nonzero.empty:
            continue

        final_time = g_nonzero["Time (s)"].max()
        final_row = g_nonzero[g_nonzero["Time (s)"] == final_time].iloc[0]

        if maxd_method == "final":
            maxd = float(final_row["Uptake_Da_mean"])

        elif maxd_method == "fit":
            maxd = fit_plateau(
                g_nonzero["Time (s)"].to_numpy(dtype=float),
                g_nonzero["Uptake_Da_mean"].to_numpy(dtype=float),
            )

        else:
            raise ValueError("maxd_method must be 'final' or 'fit'")

        # For both methods, use final-timepoint replicate SD as the practical
        # maxD uncertainty. This reflects replicate scatter at the high-D point.
        maxd_sd = float(final_row["Uptake_Da_SD"])
        n_maxd = int(final_row["n_replicates"])

        record["maxD_mean"] = maxd
        record["maxD_SD"] = maxd_sd
        record["n_maxD_replicates"] = n_maxd
        record["maxD_time_s"] = final_time

        maxd_records.append(record)

    if not maxd_records:
        raise ValueError("Could not estimate maxD values.")

    maxd_mean = pd.DataFrame(maxd_records)

    maxd_mean = maxd_mean[
        np.isfinite(maxd_mean["maxD_mean"]) & (maxd_mean["maxD_mean"] > 0)
    ].copy()

    if maxd_mean.empty:
        raise ValueError("Could not estimate valid positive maxD values.")

    abs_mean = abs_mean.merge(maxd_mean, on=identity_cols, how="left")

    # Optional absolute uptake table before fractional conversion.
    if write_absolute_csv is not None:
        abs_path = Path(write_absolute_csv)
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_mean.to_csv(abs_path, index=False)

        if verbose:
            print(f"Wrote replicate-averaged absolute uptake table to {abs_path}")

    # ------------------------------------------------------------------
    # 6. Convert to fractional uptake and propagate ratio uncertainty.
    # ------------------------------------------------------------------
    output_parts = []

    for _, g in abs_mean.groupby(identity_cols, dropna=False, sort=False):
        g = g.sort_values("Time (s)").copy()

        maxD = float(g["maxD_mean"].iloc[0])
        maxD_sd = float(g["maxD_SD"].iloc[0])

        if not np.isfinite(maxD) or maxD <= 0:
            peptide_id = tuple(g.iloc[0][peptide_cols])
            warnings.warn(f"Skipping peptide {peptide_id}: invalid maxD={maxD}")
            continue

        g["Rel. Uptake"] = g["Uptake_Da_mean"] / maxD
        g["Rel. Uptake SD"] = safe_ratio_sd(
            value_num=g["Uptake_Da_mean"].to_numpy(dtype=float),
            sd_num=g["Uptake_Da_SD"].to_numpy(dtype=float),
            value_den=maxD,
            sd_den=maxD_sd,
        )

        if not keep_time_zero:
            g = g[g["Time (s)"] > 0].copy()

        output_parts.append(
            g[
                peptide_cols
                + [
                    "Time (s)",
                    "Rel. Uptake",
                    "Rel. Uptake SD",
                ]
            ]
        )

    if not output_parts:
        raise ValueError("No fractional uptake curves were produced.")

    out = pd.concat(output_parts, ignore_index=True)

    out = out.sort_values(
        ["Start", "End", "Sequence", "Time (s)"]
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Optional pooled relative SD.
    #
    # This estimates one common SD from replicate-level fractional uptakes:
    #
    #     pooled_SD = sqrt( sum_i sum_j (x_ij - mean_i)^2 / sum_i (n_i - 1) )
    #
    # where i is a peptide/timepoint group and j is an independent
    # measurement within that group. This gives every final fractional
    # uptake datapoint the same error.
    # ------------------------------------------------------------------
    if pooled_relative_sd:
        valid_groups = abs_mean[
            identity_cols + ["Time (s)", "maxD_mean"]
        ].copy()

        valid_groups = valid_groups[
            np.isfinite(valid_groups["maxD_mean"])
            & (valid_groups["maxD_mean"] > 0)
        ].copy()

        rep_frac = deut.merge(
            valid_groups[identity_cols + ["Time (s)", "maxD_mean"]],
            on=identity_cols + ["Time (s)"],
            how="inner",
        )

        if not keep_time_zero:
            rep_frac = rep_frac[rep_frac["Time (s)"] > 0].copy()

        rep_frac["Rel_Uptake_rep"] = (
            rep_frac["Uptake_Da"] / rep_frac["maxD_mean"]
        )

        rep_frac = rep_frac[
            np.isfinite(rep_frac["Rel_Uptake_rep"])
        ].copy()

        group_cols = identity_cols + ["Time (s)"]
        group_means = (
            rep_frac.groupby(group_cols, dropna=False)["Rel_Uptake_rep"]
            .mean()
            .rename("Rel_Uptake_group_mean")
            .reset_index()
        )

        rep_frac = rep_frac.merge(group_means, on=group_cols, how="left")
        rep_frac["sq_dev"] = (
            rep_frac["Rel_Uptake_rep"] - rep_frac["Rel_Uptake_group_mean"]
        ) ** 2

        group_counts = rep_frac.groupby(group_cols, dropna=False).size()
        dof = int((group_counts - 1).sum())

        if dof <= 0:
            raise ValueError(
                "Cannot calculate pooled relative SD: need at least one "
                "peptide/timepoint group with >=2 independent measurements."
            )

        pooled_sd = float(np.sqrt(rep_frac["sq_dev"].sum() / dof))

        if not np.isfinite(pooled_sd):
            raise ValueError("Calculated pooled relative SD is not finite.")

        out["Rel. Uptake SD"] = pooled_sd

        if verbose:
            print(
                f"Using pooled relative SD = {pooled_sd:.6g} for all "
                f"{len(out)} fractional uptake rows."
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    if verbose:
        print(f"Wrote {len(out)} fractional uptake rows to {output_csv}")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert HDX-MS cluster-data CSV files to fractional uptake curves, "
            "using replicate-averaged absolute uptake and propagated maxD error."
        )
    )

    parser.add_argument(
        "input_csv",
        help="Input cluster-data CSV file",
    )

    parser.add_argument(
        "-o",
        "--output_csv",
        required=True,
        help="Output fractional uptake CSV file",
    )

    parser.add_argument(
        "--exposure-unit",
        default="ms",
        choices=["ms", "s", "min"],
        help="Unit of the input Exposure column. Default: ms",
    )

    parser.add_argument(
        "--maxd-method",
        default="final",
        choices=["final", "fit"],
        help="How to estimate maxD/plateau. Default: final",
    )

    parser.add_argument(
        "--state",
        default=None,
        help=(
            "Optional State substring to keep. "
            "Rows are retained if State contains this string, case-insensitive. "
            "If omitted, all rows are used. State is never used to define replicates; "
            "replicates are always exact File entries."
        ),
    )

    parser.add_argument(
        "--min-replicates",
        type=int,
        default=1,
        help=(
            "Discard peptide/timepoint groups with fewer than this many "
            "independent measurements. Default: 1"
        ),
    )

    parser.add_argument(
        "--keep-time-zero",
        action="store_true",
        help="Include the time-zero point in the output. Default: omit it.",
    )

    parser.add_argument(
        "--write-absolute-csv",
        default=None,
        help=(
            "Optional path to write the replicate-averaged absolute uptake table, "
            "including Uptake_Da_mean, Uptake_Da_SD, maxD_mean and maxD_SD."
        ),
    )

    parser.add_argument(
        "--pooled-relative-sd",
        action="store_true",
        help=(
            "Use one pooled standard deviation across replicate-level fractional "
            "uptakes from all peptides/timepoints, so every final fractional "
            "uptake datapoint has the same Rel. Uptake SD."
        ),
    )

    parser.add_argument(
        "--no-protein-identity",
        action="store_true",
        help=(
            "Do not include Protein as an internal peptide identity column, "
            "even if present."
        ),
    )

    parser.add_argument(
        "--no-modification-identity",
        action="store_true",
        help=(
            "Do not include Modification/PTM_ID as internal peptide identity columns, "
            "even if present."
        ),
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress diagnostic print statements.",
    )

    args = parser.parse_args()

    convert_cd_to_uptake(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        exposure_unit=args.exposure_unit,
        maxd_method=args.maxd_method,
        state=args.state,
        min_replicates=args.min_replicates,
        keep_time_zero=args.keep_time_zero,
        write_absolute_csv=args.write_absolute_csv,
        pooled_relative_sd=args.pooled_relative_sd,
        include_protein_in_identity=not args.no_protein_identity,
        include_modification_in_identity=not args.no_modification_identity,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()