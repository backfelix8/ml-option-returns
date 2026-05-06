"""
@file evaluate_effects.py
@brief Run effect evaluation of moneyness, time to maturity, time of day

This file can be used to run an evaluation of different moneyness, time to maturity and
time of day values, whilst all other features are kept at their mean, on the model prediction and
plots a matrix of these predictions (Figure 4.7)

@details
Values to set manually:
- IMPORTANT_COLUMNS - Features that we payed special attention to in some models (always the same in this thesis)
- FOLDER - Specifies the folder where the models are saved
- MODULATOR_FIRST - If the modulating network was trained first
    (only relevant for sequential models (Main-SMF, Mod-SMF))
- MODEL_TYPES - For each used model the model type. Possible ['rf', 'ffn', 'hypernet', 'autoencoder',
    'attention', 'fusion', 'fusionContextFirst', 'fusionComplex'] (Slow when including attention)
    (Needs to be same length as MODEL_NAMES, so maybe add model types multiple times)
    (Models will be combined as Ensemble, if only one model should be evaluated, give only one)
- MODEL_NAMES - Names of the models in the folder. E.g. 'ffn' means the models are named 'ffn_model_i_j.pt'
    and 'rf' means 'rf_modeli_j.pkl'. i denotes the rolling window and j the run per window
- MODEL_WEIGHTS - Weights of the models in the Ensemble model
- NORMALIZE - For each model whether it was trained with normalized features or not (trees no, rest yes)
- GRANULARITY - How granular should the plot be. Amount of steps between min and max value per feature

Functions:
- expand_row_with_steps(df, column, col_min, col_max, steps) - Takes a df with a single contract
    and makes steps contracts out of it, where a single column varies linearly from min to max
- create_plot(feat1, feat2, normalizations, model, title) - Plots a matrix of feature effects for feat1
    and feat2 for given model
- run_effect_evaluation() - Runs the creation of multiple plots for different feature combinations
    (moneyness, days to maturity and time of day)
"""

import os
os.environ["MPLBACKEND"] = "Agg"

from pathlib import Path

from plot_saver import save_current_figure

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from CombinedModel import CombinedModel

#Some params for plotting
plt.rcParams["figure.figsize"] = (16,9)
plt.rcParams.update({'font.size': 14})

#These are the features we payed special attention to in some models. Always the same across the thesis
IMPORTANT_COLUMNS = ['moneyness', 'normalizedMoneyness', 'time', 'daystomaturity', 'putcall_P']
FOLDER = '../models/Ensemble' #This specifies where to find the saved model files
#True for models, where we performed a sequential optimization and used the
#modulator/contractual features first (only True for Mod-SMF model)
MODULATOR_FIRST = False

#Which model types to use. Possible ['rf', 'ffn', 'hypernet', 'autoencoder', 'attention',
# 'fusion', 'fusionComplex'] (Slow when including attention) (Needs to be same length as MODEL_NAMES,
# so maybe add model types multiple times)
MODEL_TYPES = ['rf', 'fusionComplex', 'attention']
#Names of the models per model type. I.e. 'ffn' means the models are named 'ffn_model_i_j.pt' and
# 'rf' means 'rf_modeli_j.pkl'. i denotes the rolling window and j the run per window
MODEL_NAMES = ['gbrt_standard', 'tripleNet', 'attention']
#Weights given to each of the model types during ensembling. Must sum up to 1
MODEL_WEIGHTS = [0.7563, 0.2050, 0.0387]
NORMALIZE = [False, True, True] #Whether the model was trained with normalized features or not
GRANULARITY = 100 #Granularity of resulting matrix plot

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_output_base_dir() -> Path:
    if len(MODEL_NAMES) == 1:
        return SCRIPT_DIR / MODEL_NAMES[0]
    return SCRIPT_DIR / "ensemble"


OUTPUT_BASE_DIR = resolve_output_base_dir()
FIG_DIR = OUTPUT_BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def expand_row_with_steps(df: pd.DataFrame, column: str, col_min: float, col_max: float,
                          steps: int = 1000) -> pd.DataFrame:
    """
    @brief Expands a single-row DataFrame into multiple rows, with one column varying
        linearly from col_min to col_mar.
    @param df: Initial DataFrame
    @param column: Which column to vary
    @param col_min: Minimum value to consider
    @param col_max: Maximum value to consider
    @param steps: Number of steps
    @return: Expanded DataFrame
    @raise ValueError: If initial df has not one row
    """
    if len(df) != 1:
        raise ValueError("DataFrame must contain exactly one row.")

    # create the steps
    steps_values = np.linspace(col_min, col_max, steps)
    # repeat the single row
    expanded_df = pd.concat([df] * steps, ignore_index=True)
    # replace the target column with the stepped values
    expanded_df[column] = steps_values

    return expanded_df

def create_plot(feat1: str, feat2: str, normalizations: pd.DataFrame, model: CombinedModel,
                title: list[str] | None = None) -> None:
    """
    @brief Creates a plot of feature effects for given model and features
    :param feat1: First feature to consider
    :param feat2: Second feature to consider
    :param normalizations: Normalization parameters
    :param model: Model to use
    :param title: What title to give to the plot
    """

    #Retrieve feature stats and set max and min
    stats = pd.read_excel('../analysis/summary_statistics.xlsx', index_col=0)
    feat1_min = stats.loc[feat1, 'min']
    feat1_max = stats.loc[feat1, 'max']
    feat2_min = stats.loc[feat2, 'min']
    feat2_max = stats.loc[feat2, 'max']

    for put_call in [0, 1]: #Iterate over Calls and Puts
        preds = []
        #Iterate over feature values for first feature
        for feat1_value in np.linspace(feat1_min, feat1_max, num=GRANULARITY):
            data = normalizations.iloc[[0]] #Retrieve mean values for other features
            data['putcall_P'] = put_call #Set put or call
            data[feat1] = feat1_value #Set first feature value
            #Set second feature value
            data = expand_row_with_steps(data, feat2, feat2_min, feat2_max, GRANULARITY)

            #Retrieve normalized features for models, that were learned with normalized features
            data_norm = data.copy()
            common_cols = data_norm.columns.intersection(normalizations.columns)
            data_norm[common_cols] = ((data[common_cols] - normalizations.loc['mean', common_cols]) /
                                      normalizations.loc['std', common_cols])

            #Predict using models from all three rolling windows (average over the windows)
            pred = model.predict([data_norm, data_norm, data_norm], [data, data, data])
            pred = pred.reshape(3, GRANULARITY)
            pred = pred.mean(axis=0)
            pred *= 100000 #Multiply for better visualization
            preds.append(pred)

        result = np.stack(preds, axis=1) #Stack results

        # Set dict for plot description
        names = {'moneyness': 'Moneyness', 'daystomaturity': 'Days to Maturity', 'time': 'Time of Day'}
        # 3d plot of predicted returns
        M, D = np.meshgrid(np.linspace(feat1_min, feat1_max, num=GRANULARITY),
                           np.linspace(feat2_min, feat2_max, num=GRANULARITY))
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(D, M, result, cmap='viridis')
        ax.set_xlabel(names[feat2])
        ax.set_ylabel(names[feat1])
        ax.set_zlabel('Predicted Return')
        if title:
            ax.set_title(title[put_call], size=22)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.tight_layout()
        save_current_figure(FIG_DIR, f"effects_{feat1}_{feat2}_{put_call}")

def run_effect_evaluation() -> None:
    """
    @brief Runs the effect evaluation (Figure 4.7)
    @raise ValueError: Ensemble with FFN not possible as ffn was initially trained
        differently and not retrained
    """
    #Retrieve normalization params and drop Returns
    normalizations = pd.read_excel('../analysis/normalization_params.xlsx', index_col=0)
    normalizations.drop(columns=['Returns'], inplace=True)

    # Move modulating features to the end except for ffn to match order during training
    if 'ffn' not in MODEL_TYPES:
        cols = normalizations.columns.tolist()
        for el in IMPORTANT_COLUMNS:
            cols.insert(-1, cols.pop(cols.index(el)))  # this will keep the last element as it was
        cols.insert(0, cols.pop(-1))  # thus move last element to front
        normalizations = normalizations[cols]

    elif len(MODEL_TYPES) > 1:
        # Ensemble with FFN not possible as ffn was initially trained differently and not retrained
        raise ValueError("Can't combine ffn with other models!")

    # Create CombinedModel instance that includes all individual models that are
    # combined and their information
    if MODULATOR_FIRST:  # Change what are the modulating features for Mod-SMF model
        imp_cols = [col for col in normalizations.columns if col not in IMPORTANT_COLUMNS]
        model = CombinedModel(f'{FOLDER}', 5, 3, MODEL_TYPES, MODEL_NAMES,
                                MODEL_WEIGHTS, imp_cols, NORMALIZE)
    else:
        model = CombinedModel(f'{FOLDER}', 5, 3, MODEL_TYPES, MODEL_NAMES,
                                MODEL_WEIGHTS, IMPORTANT_COLUMNS, NORMALIZE)

    #Moneyness + Time to maturity
    create_plot('moneyness', 'daystomaturity', normalizations, model, ['Call', 'Put'])
    #Moneyness + Time of Day
    create_plot('moneyness', 'time', normalizations, model)
    #Time to maturity + Time of Day
    create_plot('daystomaturity', 'time', normalizations, model)

if __name__ == "__main__":
    run_effect_evaluation()
