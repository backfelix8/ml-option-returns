"""
@file Config.py
@brief Contains train configurations

This module contains all configurations for ray.tune for the hyperparameter optimization during training.

@details
Classes:
- Config - Class to handle configurations

@package Config
"""

from ray import tune
import torch

class Config:
    """
    @brief Class to handle configurations
    @details Contains methods to retrieve and modify configuration for a specific model during training.
    """

    @staticmethod
    def get_config(model_type: str, regularization: str = 'none') -> tuple[list[dict], int]:
        """
        @brief Returns the configuration for the given model_type, adjusted for the regularization strategy
        @param model_type: model_type
        @param regularization: one of 'none' | 'l2' | 'noise' | 'discrete'
        @return: list of dictionaries containing configuration for one run each, int number of samples to tune
        """
        configs, samples = Config._get_base_config(model_type)
        return Config._apply_regularization(configs, model_type, regularization), samples

    @staticmethod
    def _apply_regularization(configs: list[dict], model_type: str, regularization: str) -> list[dict]:
        """
        @brief Overwrites regularization-related hyperparameter entries depending on the chosen strategy.
               Only modifies GBR, RF and FFN; all other models are returned unchanged.
        @param configs: list of config dicts as returned by _get_base_config
        @param model_type: used to guard which models are modified
        @param regularization: one of 'none' | 'l2' | 'noise' | 'discrete'
        @return: modified configs
        """
        if model_type not in ('GBR', 'RF', 'FFN'):
            return configs

        for c in configs:
            # --- Tree models (GBR, RF): control lambda_l2 ---
            if 'lambda_l2' in c:
                if regularization == 'none':
                    c['lambda_l2'] = 0                           # fixed off → clean baseline
                elif regularization == 'l2':
                    c['lambda_l2'] = tune.uniform(1e-4, 0.3)    # enforced > 0

            # --- FFN: weight_decay is always 0 (decoupled L2 disabled for all experiments).
            # For 'l2' we use a coupled penalty directly in the loss instead (see train.py).
            if 'weight_decay' in c:
                c['weight_decay'] = 0

            # Coupled L2 penalty coefficient tuned for 'l2' experiment (FFN only)
            if regularization == 'l2' and 'weight_decay' in c:
                c['lambda_l2'] = tune.loguniform(1e-6, 1e-2)

            # Noise std tuned for 'noise' experiment (FFN only)
            if regularization == 'noise' and 'weight_decay' in c:
                c['noise_std'] = tune.loguniform(1e-6, 1e-3)

        return configs

    @staticmethod
    def _get_base_config(model_type: str) -> tuple[list[dict], int]:
        """
        @brief Returns the base configuration for the given model_type (unmodified search spaces)
        @param model_type: model_type
        @return: list of dictionaries containing configuration for one run each, int number of samples to tune
        @exception ValueError: Raises ValueError if model_type is invalid
        """
        if model_type == 'TransformerModel': #Att
            return [{
            'lr': tune.loguniform(0.0001, 0.01),
            'num_layers_decoder': 1,
            'd_model': tune.choice([1, 2]), #dimensionality (for Embedding)
            'weight_decay': tune.uniform(0, 0.1),
            'dim_feedforward_transformer': tune.choice([2, 4, 8]), #feedforward insider Transformer blocks
            'layers_feedforward': tune.choice([1, 2, 3]), #feedforward after Transformer
            'dim_feedforward': tune.choice([2,4,8]), #feedforward after Transformer
            'dropout': tune.uniform(0, 0.5),
            'batch_size': tune.choice([512, 1024, 2048]),
            'early_stopping': 2,
            'epochs': 50,
            'amsgrad': True,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            # accumulate gradients over 8 batches before updating weights
            # (more stable and less computationally demanding)
            'accumulation_steps': 8,
            }], 20

        elif model_type == 'AutoencoderModel': #AE
            return [{
            'lr': tune.loguniform(0.0001,0.01),
            'hidden_layers_main': tune.choice([1,2,3,4,5]),
            'hidden_layers_context': tune.choice([1,2,3]),
            'hidden_layers_final': tune.choice([1,2,3]),
            'hidden_dim_main': tune.choice([8, 16, 32, 64, 128]),
            'hidden_dim_context': tune.choice([8,16,32]),
            'hidden_dim_final': tune.choice([8,16,32]),
            'factor_amount': tune.choice([1,2,3,4,5,6,7,8,9,10]),
            'weight_decay': tune.uniform(0, 0.1),
            'dropout': tune.uniform(0, 0.5),
            'early_stopping': 2,
            'epochs': 50,
            'amsgrad': True,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            }], 100

        elif model_type == 'FFN': #FFN
            return [{
            'lr': tune.loguniform(0.0001,0.01),
            'hidden_layers': tune.choice([1,2,3,4,5]),
            'hidden_dim': tune.choice([8, 16, 32, 64, 128]),
            'weight_decay': tune.uniform(0, 0.1),
            'dropout': tune.uniform(0, 0.5),
            'batch_size': tune.choice([4096, 8192, 16384]),
            'early_stopping': 5,
            'epochs': 50,
            'amsgrad': True,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            }], 100

        elif model_type == 'GBR': #GBR
            return [{
                'objective': 'regression',
                'metric': 'l2',
                'boosting_type': 'gbdt',
                'num_iterations': 500,
                'num_leaves': tune.randint(10, 251),
                'max_depth': tune.randint(3, 11),
                'learning_rate': tune.loguniform(0.001,1),
                'feature_fraction': tune.uniform(0.25, 1),
                'bagging_fraction': tune.uniform(0.25, 1),
                'bagging_freq': tune.choice([1, 5, 10]),
                'feature_penalty_0': 1, #not relevant for not adjusted variance
                'feature_penalty_1': 1,
                'feature_penalty_2': 1,
                'feature_penalty_3': 1, 
                'feature_penalty_4': 1,
                'feature_penalty_5': 1,
                'feature_penalty_6': 1, 
                'feature_penalty_7': 1,
                'feature_penalty_8': 1,
                'feature_penalty_9': 1, 
                'lambda_l1': tune.uniform(0, 0.1),
                'lambda_l2': tune.uniform(0, 0.1),
                'num_threads': 8,
                'device_type': 'gpu',
                'verbose': 2
            }], 100

        elif model_type == 'GBR-AV': #GBR-AV
            return [{
                'objective': 'regression',
                'metric': 'l2',
                'boosting_type': 'gbdt',
                'num_iterations': 500,
                'num_leaves': tune.randint(10, 251),
                'max_depth': tune.randint(3, 11),
                'learning_rate': tune.loguniform(0.001,1),
                'feature_fraction': tune.uniform(0.25, 1),
                'bagging_fraction': tune.uniform(0.25, 1),
                'bagging_freq': tune.choice([1, 5, 10]),
                # feature importance for adjusted variance (feature 1 of modulating features -> moneyness)
                'feature_penalty_0': tune.uniform(1, 10),
                'feature_penalty_1': tune.uniform(1, 10),
                'feature_penalty_2': tune.uniform(1, 10),
                'feature_penalty_3': tune.uniform(1, 10),
                'feature_penalty_4': tune.uniform(1, 10),
                'feature_penalty_5': tune.uniform(1, 10),
                'feature_penalty_6': tune.uniform(1, 10),
                'feature_penalty_7': tune.uniform(1, 10),
                'feature_penalty_8': tune.uniform(1, 10),
                'feature_penalty_9': tune.uniform(1, 10),
                'lambda_l1': tune.uniform(0, 0.1),
                'lambda_l2': tune.uniform(0, 0.1),
                'num_threads': 8,
                'device_type': 'gpu',
                'verbose': 2
            }], 100

        elif model_type == 'RF': #RF
            return [{
                'objective': 'regression',
                'metric': 'l2',
                'boosting_type': 'rf',
                'num_iterations': 500,
                'num_leaves': tune.randint(10, 251),
                'max_depth': tune.randint(3, 11),
                'learning_rate': tune.loguniform(0.001,1),
                'feature_fraction': tune.uniform(0.25, 1),
                'bagging_fraction': tune.uniform(0.25, 1),
                'bagging_freq': tune.choice([1, 5, 10]),
                'feature_penalty_0': 1, #not relevant for not adjusted variance
                'feature_penalty_1': 1,
                'feature_penalty_2': 1,
                'feature_penalty_3': 1, 
                'feature_penalty_4': 1,
                'feature_penalty_5': 1,
                'feature_penalty_6': 1, 
                'feature_penalty_7': 1,
                'feature_penalty_8': 1,
                'feature_penalty_9': 1, 
                'lambda_l1': tune.uniform(0, 0.1),
                'lambda_l2': tune.uniform(0, 0.1),
                'num_threads': 8,
                'device_type': 'gpu',
                'verbose': 2
            }], 100

        elif model_type == 'RF-AV': #RF-AV
            return [{
                'objective': 'regression',
                'metric': 'l2',
                'boosting_type': 'rf',
                'num_iterations': 500,
                'num_leaves': tune.randint(10, 251),
                'max_depth': tune.randint(3, 11),
                'learning_rate': tune.loguniform(0.001,1),
                'feature_fraction': tune.uniform(0.25, 1),
                'bagging_fraction': tune.uniform(0.25, 1),
                'bagging_freq': tune.choice([1, 5, 10]),
                # feature importance for adjusted variance (feature 1 of modulating features -> moneyness)
                'feature_penalty_0': tune.uniform(1, 10),
                'feature_penalty_1': tune.uniform(1, 10),
                'feature_penalty_2': tune.uniform(1, 10),
                'feature_penalty_3': tune.uniform(1, 10),
                'feature_penalty_4': tune.uniform(1, 10),
                'feature_penalty_5': tune.uniform(1, 10),
                'feature_penalty_6': tune.uniform(1, 10),
                'feature_penalty_7': tune.uniform(1, 10),
                'feature_penalty_8': tune.uniform(1, 10),
                'feature_penalty_9': tune.uniform(1, 10),   
                'lambda_l1': tune.uniform(0, 0.1),
                'lambda_l2': tune.uniform(0, 0.1),
                'num_threads': 8,
                'device_type': 'gpu',
                'verbose': 2
            }], 100

        elif model_type == 'Hypernetwork': #Hyp
            return [{
            'lr': tune.loguniform(0.0001,0.01),
            'hidden_layers_main': tune.choice([1,2,3,4,5]),
            'hidden_dim_main': tune.choice([8, 16, 32, 64, 128]),
            'hidden_layers_hyper': tune.choice([1, 2, 3]),
            'hidden_dim_hyper': tune.choice([8, 16, 32]),
            'weight_decay': tune.uniform(0, 0.1),
            'dropout': tune.uniform(0, 0.5),
            'batch_size': tune.choice([4096, 8192, 16384]),
            'early_stopping': 5,
            'epochs': 50,
            'amsgrad': True,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            }], 100

        elif model_type == 'DoubleNet': #JSMF
            return [{
            'lr': tune.loguniform(0.0001,0.01),
            'hidden_layers_main': tune.choice([1,2,3,4,5]),
            'hidden_layers_context': tune.choice([1,2,3]),
            'hidden_dim_main': tune.choice([8, 16, 32, 64, 128]),
            'hidden_dim_context': tune.choice([8,16,32]),
            'weight_decay': tune.uniform(0, 0.1),
            'dropout': tune.uniform(0, 0.5),
            'batch_size': tune.choice([4096,8192,16384]),
            'early_stopping': 5,
            'epochs': 50,
            'amsgrad': True,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            }], 100

        elif model_type == 'TripleNet': #JSRF
            return [{
            'lr': tune.loguniform(0.0001,0.01),
            'hidden_layers_main': tune.choice([1,2,3,4,5]),
            'hidden_layers_context': tune.choice([1,2,3]),
            'hidden_layers_final': tune.choice([1,2,3]),
            'hidden_dim_main': tune.choice([8, 16, 32, 64, 128]),
            'hidden_dim_context': tune.choice([8,16,32]),
            'hidden_dim_final': tune.choice([8,16,32]),
            'weight_decay': tune.uniform(0, 0.1),
            'dropout': tune.uniform(0, 0.5),
            'batch_size': tune.choice([4096,8192,16384]),
            'early_stopping': 5,
            'epochs': 50,
            'amsgrad': True,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            }], 100

        elif model_type in ['fusion', 'fusionContextFirst']: #Main-SMF and Mod-SMF
            return [{
            'lr': tune.loguniform(0.0001,0.01),
            'hidden_layers': tune.choice([1,2,3,4,5]),
            'hidden_dim': tune.choice([8, 16, 32, 64, 128]),
            'weight_decay': tune.uniform(0, 0.1),
            'dropout': tune.uniform(0, 0.5),
            'batch_size': 8192,
            'early_stopping': 5,
            'epochs': 50,
            'amsgrad': True,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            },{
            'lr': tune.loguniform(0.0001,0.01),
            'hidden_layers': tune.choice([1,2,3]),
            'hidden_dim': tune.choice([8, 16, 32]),
            'weight_decay': tune.uniform(0, 0.1),
            'dropout': tune.uniform(0, 0.5),
            'batch_size': 8192,
            'early_stopping': 5,
            'epochs': 50,
            'amsgrad': True,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            }], 50 #Two configurations (first for most features and second for modulating features)

        else: #raise Exception if model_type not existing
            raise ValueError('Unknown model type!')
