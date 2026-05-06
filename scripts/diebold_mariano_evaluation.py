"""
@file diebold_mariano_evaluation.py
@brief Run Diebold-Mariano matrix for the rerun Jakob pipeline (panel folders such as J3/J5/J10)

Adapted for:
1) parquet data in selected_data/
2) model files stored inside per-model subfolders
3) valid normalization_params.xlsx auto-detection
4) equal-weight ensemble
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from CombinedModel import CombinedModel


# =========================================================
# 1. Basic panel settings
# =========================================================
PANEL_ROOT = Path(__file__).resolve().parent          # e.g. .../selected_data/J10
SELECTED_DATA_PATH = PANEL_ROOT.parent                # .../selected_data
ANALYSIS_DIR = SELECTED_DATA_PATH / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

TEST_MONTHS = [10, 11, 12]
N_WINDOWS = 3
N_RUNS = 5

ONLY_PUT = False
ONLY_CALL = False

if ONLY_PUT and ONLY_CALL:
    raise ValueError("ONLY_PUT and ONLY_CALL cannot both be True.")


# =========================================================
# 2. IMPORTANT: current modulation features
#    Replace only this list if your current J10 modulation set differs.
# =========================================================
IMPORTANT_COLUMNS = [
    'theta',
    'bid_size',
    'ask_size',
    'implVol',
    'vega',
    'normalizedMoneyness',
    'time',
    'Underlying_Ret_D2',
    'Underlying_Ret_H1',
    'delta'
]


# =========================================================
# 3. Model settings
# =========================================================
MODEL_TYPES = [
    'rf', 'rf', 'rf', 'rf',
    'ffn',
    'fusion', 'fusionContextFirst',
    'fusionComplex', 'fusionComplex',
    'hypernet',
    'attention',
    'autoencoder'
]

MODEL_NAMES = [
    'gbrt_standard',   # GBR
    'gbrt',            # GBR-AV
    'rf_standard',     # RF
    'rf',              # RF-AV
    'ffn',
    'fusion',
    'fusion_context_first',
    'doubleNet',
    'tripleNet',
    'hypernet',
    'attention',
    'autoencoder'
]

NORMALIZE = [
    False, False, False, False,
    True,
    True, True,
    True, True,
    True,
    True,
    True
]

# equal-weight ensemble for current paper version
ENSEMBLE_MODELS = ['gbrt_standard', 'tripleNet', 'attention']
WEIGHTS = [1/3, 1/3, 1/3]


# =========================================================
# 4. Utility functions
# =========================================================
def safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def clean_temp_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def find_valid_normalization_params(panel_root: Path) -> Path:
    """
    Choose normalization_params.xlsx only from models that actually use normalization,
    and require:
      - index contains 'mean' and 'std'
      - columns contain 'Returns'
    """
    for model_name, normalize in zip(MODEL_NAMES, NORMALIZE):
        if not normalize:
            continue

        candidate = panel_root / model_name / "normalization_params.xlsx"
        if not candidate.exists():
            continue

        try:
            df = pd.read_excel(candidate, index_col=0)
        except Exception:
            continue

        if ('mean' in df.index) and ('std' in df.index) and ('Returns' in df.columns):
            print(f"[INFO] Using normalization params from: {candidate}")
            return candidate

    raise FileNotFoundError(
        "No valid normalization_params.xlsx found in normalized model folders "
        "(must contain index ['mean','std'] and column 'Returns')."
    )


def prepare_flat_model_dir(panel_root: Path, flat_dir: Path) -> None:
    """
    CombinedModel expects all model files to sit in one flat folder.
    This function gathers all .pt/.pkl model files from per-model subfolders into a temp flat directory.
    """
    clean_temp_dir(flat_dir)

    for model_name in MODEL_NAMES:
        src_dir = panel_root / model_name
        if not src_dir.exists():
            raise FileNotFoundError(f"Missing model folder: {src_dir}")

        model_files = [
            fp for fp in src_dir.iterdir()
            if fp.is_file()
            and fp.suffix in {".pt", ".pkl"}
            and fp.name.startswith(model_name)
        ]

        if len(model_files) == 0:
            raise FileNotFoundError(f"No .pt/.pkl model files found in {src_dir}")

        for fp in model_files:
            dst = flat_dir / fp.name
            safe_link_or_copy(fp, dst)

    print(f"[INFO] Flat model directory prepared at: {flat_dir}")


def load_raw_test_months() -> list[pd.DataFrame]:
    dfs = []
    for m in TEST_MONTHS:
        fp = SELECTED_DATA_PATH / f"data_month_{m}.parquet"
        if not fp.exists():
            raise FileNotFoundError(f"Missing parquet file: {fp}")

        df = pd.read_parquet(fp)

        float_cols = df.select_dtypes(include=['float']).columns
        df[float_cols] = df[float_cols].astype(np.float32)

        int_cols = df.select_dtypes(include=['int']).columns
        df[int_cols] = df[int_cols].astype(np.int32)

        if ONLY_PUT:
            df = df[df['putcall_P'] == 1].copy()
        if ONLY_CALL:
            df = df[df['putcall_P'] == 0].copy()

        dfs.append(df)

    return dfs


def apply_saved_zscore(raw_dfs: list[pd.DataFrame], params_path: Path) -> list[pd.DataFrame]:
    params = pd.read_excel(params_path, index_col=0)

    if 'mean' not in params.index or 'std' not in params.index:
        raise ValueError(f"{params_path} does not contain 'mean'/'std' index.")
    if 'Returns' not in params.columns:
        raise ValueError(f"{params_path} does not contain 'Returns' column.")

    dfs_norm = []
    for df in raw_dfs:
        df_norm = df.copy()
        common_cols = df_norm.columns.intersection(params.columns)

        for col in common_cols:
            std = params.loc['std', col]
            if pd.notna(std) and std != 0:
                df_norm[col] = (df_norm[col] - params.loc['mean', col]) / std

        float_cols = df_norm.select_dtypes(include=['float']).columns
        df_norm[float_cols] = df_norm[float_cols].astype(np.float32)

        dfs_norm.append(df_norm)

    return dfs_norm


def preprocess_for_model(dfs: list[pd.DataFrame], keep_ffn_order: bool) -> tuple[list[pd.DataFrame], list[pd.Series]]:
    """
    Match evaluate.py/train.py inference-time preprocessing order:
    1) copy target
    2) drop metadata / next-period / id columns
    3) reorder modulators only for non-FFN branch
    4) drop Returns from features
    """
    X_tests = []
    y_tests = []

    for df_raw in dfs:
        df = df_raw.copy()
        y_test = df["Returns"].copy()

        cols_to_drop = [
            'loctimestamp', 'Unnamed: 0', 'price_nex', 'underlyingprice_nex',
            'optspread_nex', 'undspread_nex', 'delta_nex', 'instrumentid'
        ]
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

        if not keep_ffn_order:
            cols = df.columns.tolist()
            for el in IMPORTANT_COLUMNS:
                if el in cols:
                    cols.insert(-1, cols.pop(cols.index(el)))
            cols.insert(0, cols.pop(-1))
            df = df[cols]

        X_test = df.drop(columns=['Returns'])

        X_tests.append(X_test)
        y_tests.append(y_test)

    return X_tests, y_tests


def expected_feature_columns(df_raw: pd.DataFrame, keep_ffn_order: bool) -> list[str]:
    df = df_raw.copy()
    cols_to_drop = [
        'loctimestamp', 'Unnamed: 0', 'price_nex', 'underlyingprice_nex',
        'optspread_nex', 'undspread_nex', 'delta_nex', 'instrumentid'
    ]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    if not keep_ffn_order:
        cols = df.columns.tolist()
        for el in IMPORTANT_COLUMNS:
            if el in cols:
                cols.insert(-1, cols.pop(cols.index(el)))
        cols.insert(0, cols.pop(-1))
        df = df[cols]

    return df.drop(columns=['Returns']).columns.tolist()


def validate_feature_columns(
    raw_dfs: list[pd.DataFrame],
    feature_sets: list[pd.DataFrame],
    keep_ffn_order: bool,
    branch_name: str
) -> None:
    expected_columns = expected_feature_columns(raw_dfs[0], keep_ffn_order=keep_ffn_order)

    for month_idx, frame in enumerate(feature_sets, start=1):
        assert frame.columns.tolist() == expected_columns, (
            f"{branch_name} feature order mismatch for test month index {month_idx}."
        )

    assert all(
        frame.columns.tolist() == feature_sets[0].columns.tolist()
        for frame in feature_sets[1:]
    ), f"{branch_name} column order differs across test months."

    if not keep_ffn_order:
        assert len(feature_sets[0].columns) == len(expected_columns), (
            "Non-FFN branch does not match evaluate.py feature count."
        )


def validate_dm_workbook(workbook_path: Path, expected_labels: list[str]) -> None:
    expected_shape = (len(expected_labels), len(expected_labels))

    for sheet_name in ['Statistic', 'p-Value']:
        df = pd.read_excel(workbook_path, sheet_name=sheet_name, index_col=0)
        actual_index = list(df.index)
        actual_columns = list(df.columns)

        print(f"[INFO] Validating sheet '{sheet_name}' from {workbook_path}")
        print(f"[INFO] {sheet_name} shape: {df.shape}")
        print(f"[INFO] {sheet_name} index: {actual_index}")
        print(f"[INFO] {sheet_name} columns: {actual_columns}")

        if df.shape != expected_shape:
            raise ValueError(
                f"Sheet '{sheet_name}' has shape {df.shape}, expected {expected_shape}."
            )
        if actual_index != expected_labels:
            raise ValueError(
                f"Sheet '{sheet_name}' index mismatch. Expected {expected_labels}, "
                f"got {actual_index}."
            )
        if actual_columns != expected_labels:
            raise ValueError(
                f"Sheet '{sheet_name}' columns mismatch. Expected {expected_labels}, "
                f"got {actual_columns}."
            )


# =========================================================
# 5. Main run
# =========================================================
def main() -> None:
    print(f"[INFO] PANEL_ROOT = {PANEL_ROOT}")
    print(f"[INFO] SELECTED_DATA_PATH = {SELECTED_DATA_PATH}")

    flat_dir = PANEL_ROOT / "_dm_flat_models"

    # 1) prepare model files in a flat temp folder
    prepare_flat_model_dir(PANEL_ROOT, flat_dir)

    # 2) find and copy a valid normalization file to PANEL_ROOT root
    norm_src = find_valid_normalization_params(PANEL_ROOT)
    norm_dst = PANEL_ROOT / "normalization_params.xlsx"
    shutil.copy2(norm_src, norm_dst)

    check_df = pd.read_excel(norm_dst, index_col=0)
    print("[INFO] normalization_params shape:", check_df.shape)
    print("[INFO] normalization_params index:", list(check_df.index))
    print("[INFO] Has Returns:", "Returns" in check_df.columns)

    # 3) load raw parquet data
    raw_tests = load_raw_test_months()

    # 4) normalized and unnormalized versions
    raw_tests_normalized = apply_saved_zscore(raw_tests, norm_dst)

    # 5) preprocess for non-FFN and FFN using the same clean-first ordering as evaluate.py/train.py
    X_test_cleans_normalized, _ = preprocess_for_model(raw_tests_normalized, keep_ffn_order=False)
    X_test_cleans_normalized_ffn, _ = preprocess_for_model(raw_tests_normalized, keep_ffn_order=True)
    X_test_cleans, _ = preprocess_for_model(raw_tests, keep_ffn_order=False)

    validate_feature_columns(raw_tests, X_test_cleans, keep_ffn_order=False, branch_name='Non-FFN raw')
    validate_feature_columns(raw_tests_normalized, X_test_cleans_normalized,
                             keep_ffn_order=False, branch_name='Non-FFN normalized')
    validate_feature_columns(raw_tests_normalized, X_test_cleans_normalized_ffn,
                             keep_ffn_order=True, branch_name='FFN normalized')

    print("[INFO] Non-FFN normalized first 20 columns:",
          X_test_cleans_normalized[0].columns[:20].tolist())
    print("[INFO] FFN normalized first 20 columns:",
          X_test_cleans_normalized_ffn[0].columns[:20].tolist())
    print("[INFO] Non-FFN normalized column count:", len(X_test_cleans_normalized[0].columns))
    print("[INFO] FFN normalized column count:", len(X_test_cleans_normalized_ffn[0].columns))
    print("[INFO] 'Returns' absent from non-FFN features:",
          'Returns' not in X_test_cleans_normalized[0].columns)
    print("[INFO] 'Returns' absent from FFN features:",
          'Returns' not in X_test_cleans_normalized_ffn[0].columns)

    # 7) run DM matrix
    cwd0 = Path.cwd()
    try:
        # CombinedModel.predict() reads ./normalization_params.xlsx
        os.chdir(PANEL_ROOT)

        models = CombinedModel(
            str(flat_dir),
            N_RUNS,
            N_WINDOWS,
            MODEL_TYPES,
            MODEL_NAMES,
            WEIGHTS,
            IMPORTANT_COLUMNS,
            NORMALIZE
        )

        models.run_diebold_mariano_matrix(
            X_test_cleans_normalized,
            X_test_cleans_normalized_ffn,
            X_test_cleans,
            raw_tests,
            ENSEMBLE_MODELS
        )

    finally:
        os.chdir(cwd0)

    # 8) copy generated Excel back into current panel folder
    generated = ANALYSIS_DIR / "diebold_mariano.xlsx"
    if not generated.exists():
        raise FileNotFoundError(f"Expected output not found: {generated}")

    if ONLY_PUT:
        out_name = "diebold_mariano_put.xlsx"
    elif ONLY_CALL:
        out_name = "diebold_mariano_call.xlsx"
    else:
        out_name = "diebold_mariano_all.xlsx"

    final_out = PANEL_ROOT / out_name
    shutil.copy2(generated, final_out)
    validate_dm_workbook(final_out, MODEL_NAMES + ['Ensemble'])

    print(f"[OK] DM matrix saved to: {final_out}")


if __name__ == '__main__':
    main()
