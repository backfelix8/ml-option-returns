"""
@file evaluate.py
@brief Run evaluation

This file can be used to run multiple evaluations of trained models.
"""

import os
os.environ["MPLBACKEND"] = "Agg"

from pathlib import Path

import pandas as pd
import numpy as np
from CombinedModel import CombinedModel


def apply_saved_zscore(dfs, params_path='./normalization_params.xlsx'):
    params = pd.read_excel(params_path, index_col=0)
    dfs_norm = []
    for df in dfs:
        df_norm = df.copy()
        num_cols = df_norm.select_dtypes(include=['number']).columns
        for col in num_cols:
            if col in params.columns:
                std = params.loc['std', col]
                if pd.notna(std) and std > 0:
                    df_norm[col] = (df_norm[col] - params.loc['mean', col]) / std
        df_norm[num_cols] = df_norm[num_cols].astype(np.float32)
        dfs_norm.append(df_norm)
    return dfs_norm


# These are the features we payed special attention to in some models. Always the same across the thesis
IMPORTANT_COLUMNS = ['theta', 'bid_size', 'ask_size','implVol','vega','normalizedMoneyness','time','Underlying_Ret_D2','Underlying_Ret_H1','delta']

FOLDER = 'Z:/Dokumente/dev/ml-option-returns/scripts/gbrt_standard'  # This specifies where to find the saved model files
ONLY_PUT = False
ONLY_CALL = False
MODULATOR_FIRST = False
EVALUATION_TYPE ='performance'
PERFORMANCE_BASIC_ONLY = True

#Z:\Dokumente\dev\ml-option-returns\scripts\gbrt_standard\gbrt_standard_model0_0.pkl

# MODEL_TYPES = [
#   'rf', 'rf', 'rf', 'rf',
#   'ffn',
#   'fusion', 'fusionContextFirst',
#   'fusionComplex', 'fusionComplex',
#   'hypernet',
#   'attention',
#   'autoencoder'
# ]
#
# MODEL_NAMES = [
#   'gbrt_standard',   # GBR
#   'gbrt',            # GBR-AV
#   'rf_standard',     # RF
#   'rf',              # RF-AV
#   'ffn',
#   'fusion',
#   'fusion_context_first',
#   'doubleNet',
#   'tripleNet',
#   'hypernet',
#   'attention',       # TransformerModel maps to attention
#   'autoencoder'
# ]
#
# NORMALIZE = [
#   False, False, False, False,
#   True,
#   True, True,
#   True, True,
#   True,
#   True,
#   True
# ]

MODEL_TYPES = ['rf']
MODEL_NAMES = ['gbrt_standard']
MODEL_WEIGHTS = [1]
NORMALIZE = [False]

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_output_base_dir() -> Path:
    if len(MODEL_NAMES) == 1:
        return SCRIPT_DIR / MODEL_NAMES[0]
    return SCRIPT_DIR / "ensemble"


OUTPUT_BASE_DIR = resolve_output_base_dir()
FIG_DIR = OUTPUT_BASE_DIR / "shap" / "figures"
ROB_DIR = OUTPUT_BASE_DIR / "robustness"
PERF_FIG_DIR = OUTPUT_BASE_DIR / "figures"

for p in [OUTPUT_BASE_DIR, FIG_DIR, ROB_DIR, PERF_FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Data paths
SELECTED_DATA_PATH = 'Z:\Dokumente\dev\ml-option-returns\data'
FULL_DATA_PATH = 'Z:\Dokumente\dev\ml-option-returns\data'


def preprocess_for_model(full_dataset: bool, reorder_modulators: bool = True) -> tuple[list[pd.DataFrame], list[pd.Series]]:
    """
    Preprocess data for model prediction using the same feature order as training.
    """
    if full_dataset:
        months = range(1, 13)
    else:
        months = range(10, 13)  # October, November, December

    X_tests = []
    y_tests = []

    for i in months:
        df = pd.read_parquet(f'{SELECTED_DATA_PATH}/data_month_{i}.parquet')

        # Transform to float32 and int32 for memory reasons
        float_cols = df.select_dtypes(include=['float']).columns
        df[float_cols] = df[float_cols].astype(np.float32)
        int_cols = df.select_dtypes(include=['int']).columns
        df[int_cols] = df[int_cols].astype(np.int32)

        # Use only puts or only calls if wished
        if ONLY_PUT:
            df = df[df['putcall_P'] == 1]
        if ONLY_CALL:
            df = df[df['putcall_P'] == 0]

        # Copy the target variable
        y_test = df["Returns"].copy()

        # Match train.py preprocessing exactly
        df.drop(columns=['loctimestamp'], inplace=True)

        # Drop the same extra columns used in training
        cols_to_drop = ['Unnamed: 0', 'price_nex', 'underlyingprice_nex', 'optspread_nex',
                        'undspread_nex', 'delta_nex', 'instrumentid']
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

        # Move modulating features to the end, consistent with train.py
        if reorder_modulators:
            cols = df.columns.tolist()
            for el in IMPORTANT_COLUMNS:
                if el in cols:
                    cols.insert(-1, cols.pop(cols.index(el)))
            cols.insert(0, cols.pop(-1))
            df = df[cols]

        # Remove Returns because it is the target, not a feature
        X_test = df.drop(columns=['Returns'])

        X_tests.append(X_test)
        y_tests.append(y_test)

    return X_tests, y_tests


def preprocess_for_portfolio(full_dataset: bool) -> list[pd.DataFrame]:
    """
    Preprocess data for portfolio evaluation, keeping extra columns such as
    price, underlyingprice, and riskfree.
    """
    if full_dataset:
        months = range(1, 13)
    else:
        months = range(10, 13)

    X_tests = []

    for i in months:
        # Read selected_data
        df = pd.read_parquet(f'{SELECTED_DATA_PATH}/data_month_{i}.parquet')

        # Read the full dataset to restore required extra columns
        df_full = pd.read_parquet(f'{FULL_DATA_PATH}/data_month_{i}.parquet')

        missing_for_positional_copy = [
            col for col in ['price', 'underlyingprice', 'riskfree']
            if col in df_full.columns and col not in df.columns
        ]
        if missing_for_positional_copy and len(df) != len(df_full):
            raise ValueError('selected_data and full_data row counts differ; positional assignment would be unsafe')

        if 'price' in df_full.columns and 'price' not in df.columns:
            df['price'] = df_full['price'].values
        if 'underlyingprice' in df_full.columns and 'underlyingprice' not in df.columns:
            df['underlyingprice'] = df_full['underlyingprice'].values
        if 'riskfree' in df_full.columns and 'riskfree' not in df.columns:
            df['riskfree'] = df_full['riskfree'].values

        del df_full

        # Transform types
        float_cols = df.select_dtypes(include=['float']).columns
        df[float_cols] = df[float_cols].astype(np.float32)
        int_cols = df.select_dtypes(include=['int']).columns
        df[int_cols] = df[int_cols].astype(np.int32)

        if ONLY_PUT:
            df = df[df['putcall_P'] == 1].copy()
        if ONLY_CALL:
            df = df[df['putcall_P'] == 0].copy()

        # Build uniqueIdent
        df['loctimestamp'] = pd.to_datetime(df['loctimestamp'], format='%Y-%m-%d')
        df['end_date'] = df['loctimestamp'] + pd.to_timedelta(df['daystomaturity'], unit='D')
        df['end_date'] = df['end_date'].dt.strftime('%Y-%m-%d')
        df['strike_calc'] = df['normalizedMoneyness'].round(4)
        df['uniqueIdent'] = (df['putcall_P'].astype(str) + '-' + df['end_date'] + '-' +
                             df['strike_calc'].astype(str))
        df.drop(columns=['end_date', 'strike_calc'], inplace=True)

        X_tests.append(df)

    return X_tests


def run_evaluation() -> None:
    """
    Run evaluation.
    """
    print(f"[output] base dir: {OUTPUT_BASE_DIR}")
    print(f"[output] figures dir: {PERF_FIG_DIR}")
    print(f"[output] shap dir: {FIG_DIR}")
    print(f"[output] robustness dir: {ROB_DIR}")

    has_ffn = ('ffn' in MODEL_TYPES)
    has_non_ffn = any(t != 'ffn' for t in MODEL_TYPES)
    if has_ffn and has_non_ffn:
        raise ValueError("FFN must be evaluated separately because training used a different feature column order (no modulator reordering).")
    reorder_modulators = (not has_ffn)

    # Load features for model prediction using the training-time feature order
    X_test_cleans, y_tests = preprocess_for_model(
        True if EVALUATION_TYPE == 'shapley' else False, reorder_modulators=reorder_modulators)

    # Load portfolio evaluation data with the required extra columns
    X_tests_portfolio = preprocess_for_portfolio(
        True if EVALUATION_TYPE == 'shapley' else False)

    # Print feature columns for debugging
    print("Features for model prediction:")
    print(X_test_cleans[0].columns.tolist())
    print(f"Number of features: {len(X_test_cleans[0].columns)}")

    needs_norm = any(NORMALIZE)
    if needs_norm:
        X_test_cleans_norm = apply_saved_zscore(X_test_cleans, './normalization_params.xlsx')
    else:
        X_test_cleans_norm = X_test_cleans

    # Create CombinedModel instance
    if MODULATOR_FIRST:
        imp_cols = [col for col in X_test_cleans[0].columns if col not in IMPORTANT_COLUMNS]
        models = CombinedModel(f'{FOLDER}', 5, 3, MODEL_TYPES, MODEL_NAMES,
                               MODEL_WEIGHTS, imp_cols, NORMALIZE)
    else:
        models = CombinedModel(f'{FOLDER}', 5, 3, MODEL_TYPES, MODEL_NAMES,
                               MODEL_WEIGHTS, IMPORTANT_COLUMNS, NORMALIZE)

    # Perform the requested evaluation
    if EVALUATION_TYPE == 'robustness':
        models.robustness(X_test_cleans_norm, X_test_cleans, out_dir=ROB_DIR, save_table=True,
                          save_fig=True, seed=42)
        print(f"[robustness] outputs saved to: {ROB_DIR}")
    elif EVALUATION_TYPE == 'shapley':
        models.shapley_values(X_test_cleans_norm, X_test_cleans, fig_dir=FIG_DIR,
                              output_dir=OUTPUT_BASE_DIR)
    elif EVALUATION_TYPE == 'performance':
        # X_test_cleans: features for model prediction
        # X_test_cleans_norm: normalized features when required
        # X_tests_portfolio: portfolio inputs including price, riskfree, etc.
        models.performancePortfolios(
            X_test_cleans_norm,
            X_test_cleans,
            X_tests_portfolio,
            basic_only=PERFORMANCE_BASIC_ONLY,
            output_dir=OUTPUT_BASE_DIR,
            fig_dir=PERF_FIG_DIR,
        )
    else:
        raise ValueError(f"EVALUATION_TYPE {EVALUATION_TYPE} not possible!")


if __name__ == '__main__':
    run_evaluation()
