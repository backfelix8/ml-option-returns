"""
@file train.py
@brief Trains models

This is the main script to train different models.

@details
Values to set manually:
- MODEL_TO_TRAIN - Which model to train. Possible 'TransformerModel' (Att), 'AutoencoderModel' (AE),
    'FFN', 'GBR', 'GBR-AV', 'RF', 'RF-AV', 'Hypernetwork' (Hyp), 'DoubleNet' (JSMF),
    'TripleNet' (JSRF), 'fusion' (Main-SMF), 'fusionContextFirst' (Mod-SMF)
- IMPORTANT_COLUMNS - Features that we payed special attention to in some models
    (always the same in this thesis)
- NORMALIZE - Whether to normalize the data before training

Classes:
- LargeDataset - Class to handle large datasets

Functions:
- preprocess() - Conducts some more preprocessing
- normalize_z_score(df, params) - Normalization of given df and possibly given parameter
- normalize(dfs) - Normalize multiple dfs together
- train_tree_model(config, X_train, y_train, X_val, y_val) - Train a single tree-based model
- train_tree_model_optuna(trial, X_train, y_train, X_val, y_val, trial_results) - Train tree model with Optuna
- train_neural_model(config, train_path, val_path, feature_cols, target_col, step, run) - Train a single Neural Network
    based model
- hyperparameter_optimization() - Train multiple models and optimize over hyperparameter (save best 5)
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.integration.lightgbm import TuneReportCheckpointCallback
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
import tempfile
from ray.train import Checkpoint
from ray.air import session
import ray
import time
import math
from Config import Config
from CombinedFusionModel import CombinedFusionModel
from model import (TransformerModel, AutoencoderModel, MainNet, Hypernetwork, MainNetHypernetwork,
                   DoubleNet, TripleNet)
import optuna
from optuna.samplers import TPESampler

# Which model to train. Possible 'TransformerModel', 'AutoencoderModel', 'FFN', 'GBR', 'GBR-AV', 'RF',
# 'RF-AV', 'Hypernetwork', 'DoubleNet', 'TripleNet', 'fusion', 'fusionContextFirst'
MODEL_TO_TRAIN = 'GBR-AV'
# Set to an integer to force-override the default Optuna/Ray trial count.
# Set to None to keep the default returned by Config.get_config(MODEL_TO_TRAIN).
TRIALS_OVERRIDE = 20
# These are the features we payed special attention to in some models. Always the same across the thesis
IMPORTANT_COLUMNS = ['theta', 'bid_size', 'ask_size','implVol','vega','normalizedMoneyness','time','Underlying_Ret_D2','Underlying_Ret_H1','delta']
# Whether the model is trained with normalized features or not (True for Neural Network based models
# and False for tree-based models across this thesis)
NORMALIZE = False
MODEL_NAME_MAP = {
    'TransformerModel': 'attention',
    'AutoencoderModel': 'autoencoder',
    'FFN': 'ffn',
    'GBR': 'gbrt_standard',
    'GBR-AV': 'gbrt',
    'RF': 'rf_standard',
    'RF-AV': 'rf',
    'Hypernetwork': 'hypernet',
    'DoubleNet': 'doubleNet',
    'TripleNet': 'tripleNet',
    'fusion': 'fusion',
    'fusionContextFirst': 'fusion_context_first',
}
MODEL_NAME = MODEL_NAME_MAP.get(MODEL_TO_TRAIN, MODEL_TO_TRAIN)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, MODEL_NAME)


def preprocess() -> tuple[list[pd.DataFrame], pd.DataFrame | None]:
    """
    @brief Conducts some more preprocessing such as normalization of the features
    @return loaded dfs (potentially normalized) and normalization params
    """
    dfs = []
    # Iterate over months in training and validation (December not relevant as used only for testing)
    for i in range(1, 12):
        df = pd.read_parquet(
            f'/root/autodl-tmp/rerun_jakob_code/jakob-code/data/month/selected_data/data_month_{i}.parquet')
        # Transform to float32 and int32 for memory reasons
        float_cols = df.select_dtypes(include=['float']).columns
        df[float_cols] = df[float_cols].astype(np.float32)
        int_cols = df.select_dtypes(include=['int']).columns
        df[int_cols] = df[int_cols].astype(np.int32)

        # Drop additional information in data, that is not used for prediction,
        # keep loctimestamp for Autoencoder as we need it later (but will also be removed before training)
        if MODEL_TO_TRAIN != 'AutoencoderModel':
            df.drop(columns=['loctimestamp'], inplace=True)
        # Safe drop to avoid KeyError if a column is missing
        cols_to_drop = ['Unnamed: 0', 'price_nex', 'underlyingprice_nex', 'optspread_nex',
                        'undspread_nex', 'delta_nex', 'instrumentid']
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

        # Move modulating features to the end except for ffn as ffn uses all features as one input
        if MODEL_TO_TRAIN != 'FFN':
            cols = df.columns.tolist()
            for el in IMPORTANT_COLUMNS:
                cols.insert(-1, cols.pop(cols.index(el)))  # this will keep last element as it was
            cols.insert(0, cols.pop(-1))  # thus move last element to front
            df = df[cols]

        dfs.append(df)

    # Normalize all features if wished
    if NORMALIZE:
        return normalize(dfs)
    else:
        return dfs, None


def normalize_z_score(df: pd.DataFrame, params: pd.DataFrame | None = None) \
        -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    @brief This function performs normalization given the data or performs a normalization
        transformation given by params
    @param df: Data to normalize
    @param params: If given assume params to be population params (mean, variance) and normalize with them
    @return: Normalized data and normalization params, if not already given
    """
    normalized_df = df.copy()
    normalization_params = {}

    if not params:  # normalize using mean and variance in df per feature
        for column in df.select_dtypes(include='number').columns:
            col_mean = df[column].mean()
            col_std = df[column].std()
            normalization_params[column] = {'mean': col_mean, 'std': col_std}
            normalized_df[column] = (df[column] - col_mean) / col_std

    else:  # use given params (mean, variance) to normalize
        for column in df.select_dtypes(include='number').columns:
            normalized_df[column] = (df[column] - params[column]['mean']) / params[column]['std']
        return normalized_df

    return normalized_df, normalization_params


def normalize(dfs: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """
    @brief This function performs the normalization of features across a dataframe
    @param dfs: dfs in list per month
    @return: normalized dfs and normalization params
    """
    df = pd.concat(dfs)
    df, params = normalize_z_score(df)  # Normalize across all data

    normalized = []
    # As we want to return the list of dfs again, normalize each df individually using params from above
    for el in dfs:
        norm = normalize_z_score(el, params)
        normalized.append(norm)
    del dfs

    return normalized, params


def train_tree_model(config: dict, X_train: pd.DataFrame, y_train: pd.DataFrame,
                     X_val: pd.DataFrame, y_val: pd.DataFrame) -> None:
    """
    @brief This method conducts the training process of a single model for tree-based models (Ray version)
    @param config: Ray config given for this model
    @param X_train: Train data
    @param y_train: Train target data
    @param X_val: Validation data
    @param y_val: Validation target data
    """

    # Transform data to LGB dataset using up to 63 bins
    train_data = lgb.Dataset(X_train, label=y_train, params={'max_bin': 63})
    val_data = lgb.Dataset(X_val, label=y_val, params={'max_bin': 63}, reference=train_data)

    # Use feature importance 1 for all features except for modulating features
    fixed_values = [1] * (X_train.shape[1] - len(IMPORTANT_COLUMNS))
    # For modulating features use feature importance given by configuration
    feature_pens = fixed_values + [config[f'feature_penalty_{i}'] for i in range(len(IMPORTANT_COLUMNS))]

    # set params for lgb training
    lgbm_params = {
        'objective': config['objective'],
        'metric': config['metric'],
        'boosting_type': config['boosting_type'],
        'num_iterations': config['num_iterations'],
        'num_leaves': config['num_leaves'],
        'max_depth': config['max_depth'],
        'learning_rate': config['learning_rate'],
        'feature_fraction': config['feature_fraction'],
        'bagging_fraction': config['bagging_fraction'],
        'bagging_freq': config['bagging_freq'],
        'feature_penalty': feature_pens,  # apply feature importance
        'lambda_l1': config['lambda_l1'],
        'lambda_l2': config['lambda_l2'],
        'num_threads': config['num_threads'],
        'device_type': config['device_type'],
        'verbose': config['verbose'],
    }

    # Train the model, save checkpoints and apply early stopping
    model = lgb.train(lgbm_params, train_data, valid_sets=[val_data], valid_names=['eval'],
                      callbacks=[lgb.early_stopping(stopping_rounds=25),
                                 TuneReportCheckpointCallback(
                                     {
                                         "l2": "eval-l2"
                                     }, frequency=1, checkpoint_at_end=True
                                 )
                                 ], )

    # Make predictions
    y_pred = model.predict(X_val)

    # Evaluate the model on validation set and report error back to tune
    mse = mean_squared_error(y_val, y_pred)
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        path = os.path.join(temp_checkpoint_dir, "model.txt")
        model.save_model(path)
        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
        session.report({"l2": mse, 'done': True}, checkpoint=checkpoint)


def create_tree_objective(X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series,
                          trial_results: list, model_type: str):
    """
    @brief Creates Optuna objective function for tree-based models
    @param X_train: Training features
    @param y_train: Training target
    @param X_val: Validation features
    @param y_val: Validation target
    @param trial_results: List to store trial results
    @param model_type: 'GBR', 'GBR-AV', 'RF', or 'RF-AV'
    @return: Objective function for Optuna
    """

    def objective(trial):
        # Set boosting_type based on the model type
        if model_type in ['RF', 'RF-AV']:
            boosting_type = 'rf'
        else:
            boosting_type = 'gbdt'

        # Hyperparameter search space, matching the original Config.py setup
        params = {
            'objective': 'regression',
            'metric': 'l2',
            'boosting_type': boosting_type,
            'num_iterations': 500,
            'num_leaves': trial.suggest_int('num_leaves', 10, 250),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.25, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.25, 1.0),
            'bagging_freq': trial.suggest_categorical('bagging_freq', [1, 5, 10]),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 0.1),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 0.1),
            'num_threads': 8,
            'device_type': 'gpu',
            'verbose': -1,
        }

        # Feature penalty
        fixed_values = [1] * (X_train.shape[1] - len(IMPORTANT_COLUMNS))

        # For AV variants, tune feature_penalty; otherwise keep all penalties at 1
        if model_type in ['GBR-AV', 'RF-AV']:
            feature_pens = fixed_values + [
                trial.suggest_float(f'feature_penalty_{i}', 1.0, 10.0)
                for i in range(len(IMPORTANT_COLUMNS))
            ]
        else:
            feature_pens = fixed_values + [1] * len(IMPORTANT_COLUMNS)

        params['feature_penalty'] = feature_pens

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train, params={'max_bin': 63})
        val_data = lgb.Dataset(X_val, label=y_val, params={'max_bin': 63}, reference=train_data)

        # Train the model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            valid_names=['eval'],
            callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)]
        )

        # Predict and evaluate
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)

        # Store trial results
        trial_results.append({
            'trial_number': trial.number,
            'mse': mse,
            'model': model,
            'params': params.copy()
        })

        return mse

    return objective


def train_neural_model(config: dict, train_path: str, val_path: str,
                       feature_cols: list[str], target_col: str,
                       step: int, run: int) -> None:
    """
    @brief This method conducts the training process of a single model for Neural Network based models
    @param config: Ray config given for this model
    @param train_path: Training data parquet path
    @param val_path: Validation data parquet path
    @param feature_cols: Feature columns to use
    @param target_col: Target column name
    @param step: Rolling window count
    @param run: Run number for models, that train multiple individual models (Main-SMF and Mod-SMF)
    """
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    y_train = train_df[target_col]
    X_train = train_df[feature_cols]
    y_val = val_df[target_col]
    X_val = val_df[feature_cols]

    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_val = X_val.apply(pd.to_numeric, errors="coerce")
    y_train = pd.to_numeric(y_train, errors="coerce")
    y_val = pd.to_numeric(y_val, errors="coerce")
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)
    y_train = y_train.replace([np.inf, -np.inf], np.nan)
    y_val = y_val.replace([np.inf, -np.inf], np.nan)

    mask_tr = y_train.notna()
    mask_va = y_val.notna()
    X_train, y_train = X_train.loc[mask_tr], y_train.loc[mask_tr]
    X_val, y_val = X_val.loc[mask_va], y_val.loc[mask_va]
    X_train = X_train.fillna(0.0)
    X_val = X_val.fillna(0.0)

    # Create LargeDataset and loader
    train_ds = LargeDataset(X_train, y_train, config)
    val_ds = LargeDataset(X_val, y_val, config)
    train_loader = DataLoader(train_ds, batch_size=None)
    val_loader = DataLoader(val_ds, batch_size=None)

    # Create models with config params depending on which model to train
    if MODEL_TO_TRAIN == 'TransformerModel':
        main_model = TransformerModel(d_model=config['d_model'], nhead=math.ceil(config['d_model'] / 4),
                                      num_layers_decoder=config['num_layers_decoder'],
                                      dim_feedforward_transformer=config['dim_feedforward_transformer'],
                                      dim_feedforward=config['dim_feedforward'],
                                      layers_feedforward=config['layers_feedforward'],
                                      input_dim=X_train.shape[1] - len(IMPORTANT_COLUMNS),
                                      dropout=config['dropout']).to(config['device'])
    elif MODEL_TO_TRAIN == 'AutoencoderModel':
        main_model = AutoencoderModel(X_train.shape[1] - 1 - len(IMPORTANT_COLUMNS),
                                      len(IMPORTANT_COLUMNS), hidden_layers_main=config['hidden_layers_main'],
                                      hidden_layers_context=config['hidden_layers_context'],
                                      hidden_layers_final=config['hidden_layers_final'],
                                      hidden_dim_main=config['hidden_dim_main'],
                                      hidden_dim_context=config['hidden_dim_context'],
                                      hidden_dim_final=config['hidden_dim_final'],
                                      factor_amount=config['factor_amount'],
                                      dropout=config['dropout']).to(config['device'])
    elif MODEL_TO_TRAIN == 'FFN':
        main_model = MainNet(input_dim=X_train.shape[1], hidden_layers=config['hidden_layers'],
                             hidden_dim=config['hidden_dim'], dropout=config['dropout']).to(config['device'])
    elif MODEL_TO_TRAIN == 'Hypernetwork':
        hypernet = Hypernetwork(input_dim=len(IMPORTANT_COLUMNS), hidden_layers=config['hidden_layers_hyper'],
                                hidden_dim=config['hidden_dim_hyper'], output_dim=(X_train.shape[1] -
                                                                                   len(IMPORTANT_COLUMNS)) * config[
                                                                                      'hidden_dim_main'],
                                dropout=config['dropout']).to(config['device'])
        main_model = MainNetHypernetwork(hidden_layers=config['hidden_layers_main'],
                                         hidden_dim=config['hidden_dim_main'], hypernet=hypernet,
                                         dropout=config['dropout']).to(config['device'])
    elif MODEL_TO_TRAIN == 'DoubleNet':
        main_model = DoubleNet(X_train.shape[1] - len(IMPORTANT_COLUMNS), len(IMPORTANT_COLUMNS),
                               hidden_layers_main=config['hidden_layers_main'],
                               hidden_layers_context=config['hidden_layers_context'],
                               hidden_dim_main=config['hidden_dim_main'],
                               hidden_dim_context=config['hidden_dim_context'],
                               dropout=config['dropout']).to(config['device'])
    elif MODEL_TO_TRAIN == 'TripleNet':
        main_model = TripleNet(X_train.shape[1] - len(IMPORTANT_COLUMNS), len(IMPORTANT_COLUMNS),
                               hidden_layers_main=config['hidden_layers_main'],
                               hidden_layers_context=config['hidden_layers_context'],
                               hidden_layers_final=config['hidden_layers_final'],
                               hidden_dim_main=config['hidden_dim_main'],
                               hidden_dim_context=config['hidden_dim_context'],
                               hidden_dim_final=config['hidden_dim_final'],
                               dropout=config['dropout']).to(config['device'])
    elif MODEL_TO_TRAIN == 'fusion' and run == 0:  # first run of Main-SMF running main model on most features
        main_model = MainNet(X_train.shape[1] - len(IMPORTANT_COLUMNS),
                             hidden_layers=config['hidden_layers'], hidden_dim=config['hidden_dim'],
                             dropout=config['dropout']).to(config['device'])
    # second run of Main-SMF running modulating model on modulating features
    elif MODEL_TO_TRAIN == 'fusion' and run == 1:
        main_model = MainNet(len(IMPORTANT_COLUMNS), hidden_layers=config['hidden_layers'],
                             hidden_dim=config['hidden_dim'], dropout=config['dropout']).to(config['device'])
        # load main model
        first_model = CombinedFusionModel(
            os.path.join(MODEL_DIR, f"{MODEL_NAME}_model_main_{step}_"), 5, config['device']
        )
    # first run of Mod-SMF running modulating model on modulating features
    elif MODEL_TO_TRAIN == 'fusionContextFirst' and run == 0:
        main_model = MainNet(len(IMPORTANT_COLUMNS), hidden_layers=config['hidden_layers'],
                             hidden_dim=config['hidden_dim'], dropout=config['dropout']).to(config['device'])
    # second run of Main-SMF running main model on most features
    elif MODEL_TO_TRAIN == 'fusionContextFirst' and run == 1:
        main_model = MainNet(X_train.shape[1] - len(IMPORTANT_COLUMNS), hidden_layers=config['hidden_layers'],
                             hidden_dim=config['hidden_dim'], dropout=config['dropout']).to(config['device'])
        # load modulating model
        first_model = CombinedFusionModel(
            os.path.join(MODEL_DIR, f"{MODEL_NAME}_model_main_{step}_"), 5, config['device']
        )

    # compile model, set optimizer and loss
    main_model = torch.compile(main_model, backend='eager')
    optimizer = AdamW(main_model.parameters(), lr=config['lr'],
                      weight_decay=config['weight_decay'], amsgrad=config['amsgrad'])
    criterion = nn.MSELoss()

    # Early stopping parameters
    best_val_loss = float('inf')
    patience = config['early_stopping']
    best_model = None
    counter = 0

    for epoch in range(config['epochs']):  # Iterate over epochs
        main_model.train()  # Enable training mode
        l = 0
        startTime = time.time()
        optimizer.zero_grad()  # Clear gradients
        for xb, specb, yb in train_loader:  # Iterate over batches
            xb, specb, yb = (xb.to(config['device']), specb.to(config['device']),
                             yb.to(config['device']).unsqueeze(1))
            if not torch.isfinite(xb).all() or not torch.isfinite(specb).all() or not torch.isfinite(yb).all():
                continue

            # pass data to model depending on model
            if MODEL_TO_TRAIN == 'AutoencoderModel':
                pred = main_model(xb, specb, yb)
            elif MODEL_TO_TRAIN == 'FFN':
                pred = main_model(xb)
            elif MODEL_TO_TRAIN in ['fusion', 'fusionContextFirst'] and run == 0:
                pred = main_model(xb)
            elif MODEL_TO_TRAIN in ['fusion', 'fusionContextFirst'] and run == 1:
                with torch.no_grad():
                    base_pred = first_model.predict(xb)  # get prediction of previously ran model
                pred = base_pred * main_model(specb)
            else:
                pred = main_model(xb, specb)
            loss = criterion(pred, yb)  # get loss
            if not torch.isfinite(loss):
                session.report({"loss": float("inf"), "done": True})
                return

            # If attention model, accumulate gradients over multiple steps before updating weights
            if MODEL_TO_TRAIN == 'TransformerModel':
                loss = loss / config['accumulation_steps']  # loss needs to be divided over steps
                loss.backward()  # backpropagation
                if (l + 1) % config['accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(main_model.parameters(), max_norm=1.0)
                    optimizer.step()  # Update weights
                    optimizer.zero_grad()  # Clear gradients for the next accumulation cycle
            else:
                optimizer.zero_grad()  # Clear gradients
                loss.backward()  # backpropagation
                torch.nn.utils.clip_grad_norm_(main_model.parameters(), max_norm=1.0)
                optimizer.step()  # Update weights

            if l % 100 == 0:  # Report learning statistics every 100 batches
                endTime = time.time()
                print(f'Batches completed {l}; Time per batch: {(endTime - startTime) / 100}', flush=True)
                startTime = time.time()
            l += 1

        # Validation
        main_model.eval()  # set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # no gradient calculation
            # For Autoencoder run training data once trough model to get factors and save them
            # (final model will have the most recent factors saved)
            if MODEL_TO_TRAIN == 'AutoencoderModel':
                main_model.initFactorList()
                for xb, specb, yb in train_loader:
                    xb, specb, yb = (xb.to(config['device']), specb.to(config['device']),
                                     yb.to(config['device']).unsqueeze(1))
                    main_model.saveFactors(xb, specb, yb)
                main_model.averageFactors()  # average factors over data and save them to model

            # Iterate over validation batches
            for xb, specb, yb in val_loader:
                xb, specb, yb = (xb.to(config['device']), specb.to(config['device']),
                                 yb.to(config['device']).unsqueeze(1))
                if not torch.isfinite(xb).all() or not torch.isfinite(specb).all() or not torch.isfinite(yb).all():
                    continue

                # pass data through model depending on model
                if MODEL_TO_TRAIN == 'AutoencoderModel':
                    pred = main_model.validate(xb, specb)
                    batchs = xb.size(0)
                elif MODEL_TO_TRAIN == 'FFN':
                    pred = main_model(xb)
                    batchs = xb.size(0)
                elif MODEL_TO_TRAIN in ['fusion', 'fusionContextFirst'] and run == 0:
                    pred = main_model(xb)
                    batchs = xb.size(0)
                elif MODEL_TO_TRAIN in ['fusion', 'fusionContextFirst'] and run == 1:
                    base_pred = first_model.predict(xb)  # get prediction of previously learned model first
                    pred = base_pred * main_model(specb)
                    batchs = specb.size(0)
                else:
                    pred = main_model(xb, specb)
                    batchs = xb.size(0)

                # Sum validation loss
                batch_loss = criterion(pred, yb)
                if not torch.isfinite(batch_loss):
                    session.report({"loss": float("inf"), "done": True})
                    return
                val_loss += (batch_loss.item() * batchs)

        # Get final validation loss and print info
        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}")

        # As long model improves slightly compared to best model set stopping counter to zero
        # and best_model to current model
        if val_loss < best_val_loss * 0.999:
            best_val_loss = val_loss
            best_model = main_model
            counter = 0
        else:  # If model not better than best model increase stopping counter
            counter += 1
            if counter >= patience:  # If patience is reaches apply early stopping and stop training
                print("Early stopping")
                break

        # Save checkpoint and report loss back to tune
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                main_model, path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            session.report({"loss": val_loss}, checkpoint=checkpoint)

    # Save checkpoint and report loss back to tune (training finished)
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
        torch.save(
            best_model, path
        )
        checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
        session.report({"loss": best_val_loss, 'done': True}, checkpoint=checkpoint)


def hyperparameter_optimization() -> None:
    """
    @brief Train multiple models and optimize over hyperparameter (save best 5)
    """
    # Set device to GPU if available (highly recommended for speed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(torch.version.cuda)
    print(device)
    print(torch.cuda.get_device_name(0))

    # set loss metric identifier for tree-based and Neural Network based models
    if MODEL_TO_TRAIN in ['GBR', 'GBR-AV', 'RF', 'RF-AV']:
        loss_metric = 'l2'
    else:
        loss_metric = 'loss'

    os.makedirs(MODEL_DIR, exist_ok=True)

    dfs, params = preprocess()  # retrieve preprocessed data
    pd.DataFrame(params).to_excel(
        'normalization_params.xlsx')  # save normalization parameter (empty, if no normalization is done)

    for step in range(0, 3):  # Iterate over rolling windows
        print(f"\n{'=' * 60}")
        print(f"Rolling Window Step {step}")
        print(f"{'=' * 60}")

        # Retrieve training and validation data for rolling window (training grows by one each rolling window)
        X_train = pd.concat(dfs[:8 + step])
        X_val = pd.concat(dfs[8 + step:9 + step])

        # Get target variable and separate from features
        y_train = X_train['Returns']
        y_val = X_val['Returns']
        X_train.drop(columns=['Returns'], inplace=True)
        X_val.drop(columns=['Returns'], inplace=True)

        print(f"Training samples: {len(X_train):,}")
        print(f"Validation samples: {len(X_val):,}")
        print(f"Features: {X_train.shape[1]}")

        feature_cols = list(X_train.columns)
        target_col = "Returns"

        # Retrieve configuration and sample amount for respective model
        config, samples_model = Config.get_config(MODEL_TO_TRAIN)
        default_samples_model = samples_model
        if TRIALS_OVERRIDE is not None:
            samples_model = TRIALS_OVERRIDE

        # Set run amount for training (only not 1 for Main-SMF and Mod-SMF)
        if MODEL_TO_TRAIN in ['fusion', 'fusionContextFirst']:
            multiple_runs = 2
            if MODEL_TO_TRAIN == 'fusionContextFirst':  # Reverse config order for Mod-SMF
                config.reverse()
        else:
            multiple_runs = 1

        for run in range(multiple_runs):  # Iterate over runs
            cache_dir = os.path.join(os.getcwd(), "tune_data_cache")
            os.makedirs(cache_dir, exist_ok=True)
            train_path = os.path.join(cache_dir, f"train_step{step}_run{run}.parquet")
            val_path = os.path.join(cache_dir, f"val_step{step}_run{run}.parquet")

            train_df = X_train.copy()
            train_df["Returns"] = y_train.values
            val_df = X_val.copy()
            val_df["Returns"] = y_val.values

            train_df.to_parquet(train_path, index=False)
            val_df.to_parquet(val_path, index=False)

            del train_df, val_df

            # ============== Tree models use Optuna to avoid Ray data size limits ==============
            if MODEL_TO_TRAIN in ['GBR', 'GBR-AV', 'RF', 'RF-AV']:
                override_note = (
                    f"override from Config default {default_samples_model}"
                    if TRIALS_OVERRIDE is not None else "using Config default"
                )
                print(
                    f"\nStarting Optuna hyperparameter optimization "
                    f"(actual trials={samples_model}; {override_note})..."
                )

                # Store results from all trials
                trial_results = []

                # Create the Optuna study
                sampler = TPESampler(seed=42 + step)
                study = optuna.create_study(
                    direction='minimize',
                    sampler=sampler,
                    study_name=f'{MODEL_TO_TRAIN}_step_{step}'
                )

                # Create the objective function
                objective = create_tree_objective(X_train, y_train, X_val, y_val,
                                                  trial_results, MODEL_TO_TRAIN)

                # Run hyperparameter optimization
                start_time = time.time()
                study.optimize(
                    objective,
                    n_trials=samples_model,
                    show_progress_bar=True,
                    gc_after_trial=True
                )
                elapsed_time = time.time() - start_time

                print(f"\nOptimization completed in {elapsed_time / 60:.1f} minutes")
                print(f"Best hyperparameters found were: {study.best_params}")
                print(f"Best l2 metrics found were: {study.best_value:.6f}")

                # Save hyperparameter results to Excel
                df_results = study.trials_dataframe()
                df_results = df_results.sort_values('value')
                df_results.to_excel(f'hyperparams_{step}.xlsx', index=False)

                # Sort by MSE and save the best 5 models
                sorted_results = sorted(trial_results, key=lambda x: x['mse'])

                print(f"\nSaving top 5 models...")
                for i in range(min(5, len(sorted_results))):
                    model = sorted_results[i]['model']
                    mse = sorted_results[i]['mse']
                    model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_model{step}_{i}.pkl")
                    joblib.dump(model, model_path)
                    print(f"  Saved {model_path} (MSE: {mse:.6f})")

            # ============== Neural network models continue to use Ray ==============
            else:
                # Initialize ray tune
                ray.shutdown()
                time.sleep(5)
                ray.init(local_mode=True)

                tuner = tune.Tuner(
                    tune.with_resources(
                        tune.with_parameters(
                            train_neural_model,
                            train_path=train_path,
                            val_path=val_path,
                            feature_cols=feature_cols,
                            target_col=target_col,
                            step=step,
                            run=run
                        ),
                        resources={"cpu": 4, "gpu": 1}),
                    tune_config=tune.TuneConfig(
                        metric=loss_metric,
                        mode="min",
                        scheduler=ASHAScheduler(max_t=config[run]['epochs']),
                        search_alg=OptunaSearch(),
                        num_samples=samples_model,
                        max_concurrent_trials=1,
                    ),
                    param_space=config[run],
                )

                # Run optimization
                results = tuner.fit()

                print(f"Best main hyperparameters found were: {results.get_best_result().config}")
                print(f"Best main metrics found were: {results.get_best_result().metrics}")
                print(f"Best main path found were: {results.get_best_result().path}")

                # Set name tags if multiple runs
                if MODEL_TO_TRAIN in ['fusion', 'fusionContextFirst']:
                    if run == 0:
                        multiple_models = '_main'
                    else:
                        multiple_models = '_adj'
                else:
                    multiple_models = ''

                # Get hyperparam results and save to excel
                df_results = results.get_dataframe()
                df_results.sort_values(by=loss_metric, inplace=True)
                df_results.to_excel(f'hyperparams_{multiple_models}{step}.xlsx')

                # Get results and sort by validation loss
                all_results = list(results)
                for el in all_results:
                    if not loss_metric in el.metrics:
                        el.metrics[loss_metric] = float('inf')
                sorted_results = sorted(all_results, key=lambda r: r.metrics[loss_metric])

                # save 5 best models by loading their last checkpoints
                for i in range(5):
                    with sorted_results[i].checkpoint.as_directory() as checkpoint_dir:
                        model = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"), weights_only=False)
                    if MODEL_TO_TRAIN in ['fusion', 'fusionContextFirst']:
                        model_filename = f"{MODEL_NAME}_model{multiple_models}_{step}_{str(i)}.pt"
                    else:
                        model_filename = f"{MODEL_NAME}_model_{step}_{str(i)}.pt"
                    torch.save(model, os.path.join(MODEL_DIR, model_filename))


class LargeDataset(IterableDataset):
    """
    @brief This class incorporates an iterable datasat to process large data in chunks
        (only used for neural network based architectures)
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, config: dict) -> None:
        """
        @brief Initialization dataset
        @param X: Input data
        @param y: Target data
        @param config: Tune configuration
        """
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)

        # Set batch size, unless Autoencoder
        if MODEL_TO_TRAIN != 'AutoencoderModel':
            self.batch_size = config['batch_size']
        # If Autoencoder group timepoints and use each timepoint as one batch (relevant to get factors properly)
        else:
            groups = list(self.X.groupby(['loctimestamp', 'time']).groups.keys())
            self.indexGroup = []
            for (timestamp, time) in groups:
                indices = self.X[(self.X['loctimestamp'] == timestamp) & (self.X['time'] == time)].index
                self.indexGroup.append(indices)
            # delete loctimestamp from data as it is not used for prediction
            self.X.drop(columns=['loctimestamp'], inplace=True)

    def process_chunk(self, chunk_x: pd.DataFrame, chunk_y: pd.Series) \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        @brief Process a chunk of data
        @param chunk_x: Input chunk
        @param chunk_y: Target chunk
        @return: processed Chunk of data into data for main net and data for modulating net
        """
        # For simple FFN net we use all data in one chunk, thus return X twice (second one not used)
        if MODEL_TO_TRAIN == 'FFN':
            X = torch.tensor(chunk_x.values, dtype=torch.float32)
            y = torch.tensor(chunk_y.values, dtype=torch.float32)
            return X, X, y

        # Get chunks for main and modulating features
        chunk_x_main = chunk_x[[col for col in chunk_x.columns if col not in IMPORTANT_COLUMNS]]
        chunk_x_context = chunk_x[IMPORTANT_COLUMNS]
        X = torch.tensor(chunk_x_main.values, dtype=torch.float32)
        specX = torch.tensor(chunk_x_context.values, dtype=torch.float32)

        # Extend dimension for Attention network for Embedding
        if MODEL_TO_TRAIN == 'TransformerModel':
            X = X.unsqueeze(-1)
            specX = specX.unsqueeze(-1)

        # Extract target variable
        y = torch.tensor(chunk_y.values, dtype=torch.float32)
        return X, specX, y

    def __iter__(self):
        """
        @brief Iterator
        """
        if MODEL_TO_TRAIN == 'AutoencoderModel':  # for Autoencoder model give back chunks for each timepoint
            for indices in self.indexGroup:
                chunk_x = self.X.loc[indices]
                chunk_y = self.y.loc[indices]
                X, specX, y = self.process_chunk(chunk_x, chunk_y)
                yield X, specX, y
        else:
            for start in range(0, len(self.X),
                               self.batch_size):  # iterate through data and retrieve chunks in batch size
                end = start + self.batch_size
                X, specX, y = self.process_chunk(self.X.iloc[start:end], self.y.iloc[start:end])
                if MODEL_TO_TRAIN == 'fusionContextFirst':  # If Mod-SMF give modulating features first back
                    yield specX, X, y
                else:
                    yield X, specX, y  # Training data

    def __len__(self):
        """
        @brief Gets length of dataset
        """
        return len(self.X)


if __name__ == "__main__":
    # Set the Optuna logging level
    optuna.logging.set_verbosity(optuna.logging.INFO)

    hyperparameter_optimization()
