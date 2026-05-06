"""
@file CombinedModel.py
@brief Module to combine individual models per run to calculate ensemble forecasts and model performance

This module is the main file to analyze the models. Module to combine individual models per run to
calculate ensemble forecasts and model performance. This module enables the creation of almost
(apart from Diebold-Mariano test matrix) all evaluations for the thesis.
The main functions in this file are called by other files (evaluate.py).

@details
Classes:
- LargeDataset - Class to handle large datasets
- CombinedModel - Class containing a model combined of individual runs
- PyTorchWrapper - Wrapper for pytorch models for SHAP calculation

@package CombinedModel
"""

try:
    import joblib
except ModuleNotFoundError:  # minimal fallback for environments without joblib installed
    import pickle

    class _JoblibCompat:
        @staticmethod
        def load(filename):
            with open(filename, "rb") as f:
                return pickle.load(f)

        @staticmethod
        def dump(obj, filename):
            with open(filename, "wb") as f:
                pickle.dump(obj, f)

    joblib = _JoblibCompat()
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy import stats
try:
    import statsmodels.api as sm
except ModuleNotFoundError:
    sm = None
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import MaxNLocator
try:
    import lightgbm as lgb
except ModuleNotFoundError:
    lgb = None
import numpy as np
from pathlib import Path
try:
    import torch
    from torch.utils.data import DataLoader, IterableDataset
except ModuleNotFoundError:
    from types import SimpleNamespace
    torch = SimpleNamespace(Tensor=object, nn=SimpleNamespace(Module=object))
    DataLoader = None
    IterableDataset = object
try:
    import shap
except ModuleNotFoundError:
    shap = None
try:
    from CombinedFusionModel import CombinedFusionModel
except ModuleNotFoundError:
    CombinedFusionModel = None
from plot_saver import save_current_figure

#Some params for plotting
plt.rcParams["figure.figsize"] = (16,9)
plt.rcParams.update({'font.size': 22})


def _resolve_output_dir(output_dir) -> Path:
    resolved = Path(output_dir) if output_dir is not None else Path.cwd()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _resolve_figure_dir(fig_dir=None, output_dir=None, subdir: str = "figures") -> Path:
    if fig_dir is not None:
        resolved = Path(fig_dir)
    else:
        resolved = _resolve_output_dir(output_dir) / subdir
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _infer_output_dir_from_fig_dir(fig_dir) -> Path:
    fig_path = Path(fig_dir)
    if fig_path.name == "figures" and fig_path.parent.name == "shap":
        return fig_path.parent.parent
    if fig_path.name == "figures":
        return fig_path.parent
    return fig_path


def _save_fig(fig_dir, name: str, dpi: int = 300):
    return save_current_figure(fig_dir, name, dpi=dpi, close=True)

class LargeDataset(IterableDataset):
    """
    @brief Class to handle large datasets
    @details This class incorporates an iterable dataset to process large data in chunks
        (only used for Neural Network based architectures).
    """
    def __init__(self, X: pd.DataFrame, important_columns: list[str], model_type: str,
                 oneBatch: bool = False) -> None:
        """
        @brief Initialize the dataset.
        @param X: Data as df
        @param important_columns: Names of modulating columns
        @param model_type: Type of model used
        @param oneBatch: Whether to return data as one batch or in smaller batches
        """
        self.X = X
        self.model_type = model_type
        self.important_columns = important_columns

        #Set batch size (bigger=better for inference, but smaller for attention due to memory)
        if self.model_type == 'attention':
            self.batch_size = 4096
        else:
            self.batch_size = 262144
        if oneBatch:
            self.batch_size = len(X)

    def process_chunk(self, chunk_x: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """
        @brief Process a chunk of data
        @param chunk_x: Chunk of data
        @return: Processed chunk of data into data for main net and data for modulating net
        """
        #main data
        chunk_x_main = chunk_x[[col for col in chunk_x.columns if col not in self.important_columns]]
        #modulating data
        chunk_x_context = chunk_x[self.important_columns]

        #create tensors
        if self.model_type == 'attention': #attention net requires third dimension for embedding
            X = torch.tensor(chunk_x_main.values, dtype=torch.float32).unsqueeze(-1)
            specX = torch.tensor(chunk_x_context.values, dtype=torch.float32).unsqueeze(-1)
        elif self.model_type == 'ffn': #ffn uses all data in one dataset, thus specx is redundant
            X = torch.tensor(chunk_x.values, dtype=torch.float32)
            specX = X
        else:
            X = torch.tensor(chunk_x_main.values, dtype=torch.float32)
            specX = torch.tensor(chunk_x_context.values, dtype=torch.float32)
        return X, specX

    def __iter__(self):
        """
        @brief Iterator
        """
        #iterate through data and retrieve chunks in batch size
        for start in range(0, len(self.X), self.batch_size):
            end = start + self.batch_size
            X, specX = self.process_chunk(self.X.iloc[start:end])
            yield X, specX

    def __len__(self):
        """
        @brief Gets length of dataset
        """
        return len(self.X)

class CombinedModel:
    """
    @brief This class combines multiple individual models to an ensemble
    @details Class to combine individual models per run to calculate ensemble forecasts and model performance.
        This class enables the creation of almost (apart from Diebold-Mariano test matrix) all evaluations
        for the thesis.
    """
    def __init__(self, model_name: str, amount: int, N: int, model_types: list[str],
                 model_names: list[str], model_weights: list[float], important_columns: list[str],
                 normalize: list[bool]) -> None:
        """
        @brief Initialize the combined model
        @param model_name: Folder with the saved model files
        @param amount: How many individual models per rolling window (we used 5 in general)
        @param N: How many rolling windows (3 for testing and evaluation)
        @param model_types: Types of models as list
        @param model_names: Names of models as list. E.g. 'ffn' means the models are named
            'ffn_model_i_j.pt' and 'rf' means 'rf_modeli_j.pkl'. i denotes the rolling window and
            j the run per window
        @param model_weights: Weights of models in ensemble as list
        @param important_columns: Names of modulating columns
        @param normalize: Whether to normalized data was used for training or not
        """
        self.model_name = model_name
        self.amount = amount
        self.N = N
        self.model_types = model_types
        self.model_names = model_names
        self.model_weights = model_weights
        self.important_columns = important_columns
        self.normalize = normalize

        #Save all models in a single three dimensional list: The first dimension determines which
        #model type (dim=1 unless we evaluate the Ensemble model)
        #The second dimension determines the rolling window (dim=3 in general) and the third the
        #individual model (dim=5 in general)
        self.totalModels = []
        for count, current_model in enumerate(self.model_names): #Iterate over model types
            allModels = []
            for i in range(self.N): #Iterate over rolling window
                #When using a sequential fusion model (Main-SMF and Mod-SMF) we load a CombinedFusionModel
                #that already combines the first trained main models
                if self.model_types[count] == 'fusion' or self.model_types[count] == 'fusionContextFirst':
                    main_model = CombinedFusionModel(self.model_name + '/' + self.model_names[count] +
                                                     '_model_main_' + str(i) + "_", amount, 'cuda')
                models = []
                for j in range(self.amount): #Iterate over individual models
                    if self.model_types[count] == 'rf': #load tree-based models
                        model = joblib.load(self.model_name + '/' + self.model_names[count] + '_model' +
                                            str(i) + '_' + str(j) + '.pkl')
                        models.append(model)
                    #Combine main and adjusting model as list and append it
                    elif (self.model_types[count] == 'fusion' or
                          self.model_types[count] == 'fusionContextFirst'):
                        adj_model = torch.load(self.model_name + '/' + self.model_names[count] +
                                               '_model_adj_' + str(i) + '_' + str(j) + '.pt',
                                               map_location='cuda', weights_only=False)
                        adj_model.eval()
                        models.append([main_model, adj_model])
                    else: #load all other models
                        model = torch.load(self.model_name + '/' + self.model_names[count] +
                                           '_model_' + str(i) + '_' + str(j) + '.pt',
                                           map_location='cuda', weights_only=False)
                        model.eval()
                        models.append(model)
                allModels.append(models)
            self.totalModels.append(allModels)

    def _forward_single_model(self, model_entry, model_type: str, x_first, x_last):
        """
        @brief Apply a single loaded model entry to one batch
        @param model_entry: Stored model entry for one run
        @param model_type: Type of the stored model
        @param x_first: Main features
        @param x_last: Context or modulating features
        @return: Model prediction tensor
        """
        if model_type == 'autoencoder':
            return model_entry.validate(x_first, x_last)
        if model_type in ['fusion', 'fusionContextFirst']:
            if (not isinstance(model_entry, (list, tuple)) or len(model_entry) != 2 or
                    not hasattr(model_entry[0], 'predict') or not callable(model_entry[1])):
                raise TypeError(
                    f"Expected fusion-style model entry [main_model, adj_model] for "
                    f"model_type={model_type}, got {type(model_entry)}"
                )
            if model_type == 'fusion':
                return model_entry[0].predict(x_first) * model_entry[1](x_last)
            return model_entry[0].predict(x_last) * model_entry[1](x_first)
        if model_type == 'ffn':
            return model_entry(x_first)
        return model_entry(x_first, x_last)

    def predict(self, xs_normalized: list[pd.DataFrame], xs: list[pd.DataFrame],
                xs_normalized_ffn: list[pd.DataFrame] | None = None,
                ensemble: list[str] | None = None) -> pd.Series | list[np.ndarray]:
        """
        @brief This function is used to perform a forward pass of the entire CombinedModel
        @param xs_normalized: Data to pass through normalized
        @param xs: Data to pass through
        @param xs_normalized_ffn: Normalized data for ffn when calculating diebold_mariano test
        @param ensemble: Ensemble model when calculating diebold_mariano test
        @return: Output of the model
        """
        allReturns = []
        for count, current_model in enumerate(self.model_names): #Iterate over each option type
            if self.normalize[count]: #Use normalized data if applicable for option type
                #Use normalized data for ffn only if ffn and diebold_mariano test
                if xs_normalized_ffn and self.model_types[count] == 'ffn':
                    x = xs_normalized_ffn
                else:
                    x = xs_normalized
            else:
                x = xs
            Ys = []
            for i in range(self.N): #Iterate over rolling windows
                if self.model_types[count] == 'rf': #Predict for tree-based methods
                    Y = self.totalModels[count][i][0].predict(x[i])
                    for j in range(1, self.amount): #Iterate over individual models and sum prediction
                        y = self.totalModels[count][i][j].predict(x[i])
                        Y += y
                    Y = Y / self.amount #average prediction over individual models in rolling window
                else: #Predict for neural network based methods
                    #Create LargeDataset
                    ds = LargeDataset(x[i], self.important_columns, self.model_types[count])
                    loader = DataLoader(ds, batch_size=None)
                    y_preds = []
                    # Iterate through batches (main features, modulating features) in LargeDataset
                    for x_first, x_last in loader:
                        x_first, x_last = x_first.to('cuda'), x_last.to('cuda')
                        Y = self._forward_single_model(
                            self.totalModels[count][i][0], self.model_types[count], x_first, x_last
                        )
                        Y = Y.detach().cpu().numpy()
                        #Do prediction for other 4 models
                        for j in range(1, self.amount):
                            y = self._forward_single_model(
                                self.totalModels[count][i][j], self.model_types[count], x_first, x_last
                            )
                            y = y.detach().cpu().numpy()
                            Y += y #Sum predictions of individual models
                        Y = Y / self.amount #average prediction over individual models
                        y_preds.append(Y)
                    Y = np.concatenate(y_preds) #concat predictions from different batches
                Ys.append(Y)
            Y = np.concatenate(Ys) #concat predictions for rolling windows
            if self.normalize[count]: #unnormalize returns if model was trained with normalized data
                normalizations = pd.read_excel('./normalization_params.xlsx', index_col=0)
                Y = (Y * normalizations.loc['std', 'Returns']) + normalizations.loc['mean', 'Returns']
                Y = Y.flatten()
            allReturns.append(Y)

        #Average returns over model types using the respective weights
        averagedReturn = 0
        # If diebold-mariano test only don't use all models for ensemble, but only specific and return both
        if ensemble:
            for i in range(len(ensemble)):
                averagedReturn += (self.model_weights[i] * allReturns[self.model_names.index(ensemble[i])])
            return allReturns + [averagedReturn]
        for i in range(len(self.model_names)):
            averagedReturn += (self.model_weights[i] * allReturns[i])
        return averagedReturn

    def evaluate(self, xs_normalized: list[pd.DataFrame], xs: list[pd.DataFrame],
                 x_long: list[pd.DataFrame], fig_dir=None) -> pd.DataFrame:
        """
        @brief This function performs the statistical evaluation of the individual and combined models
            (Sheet models in performance.xlsx)
        @param xs_normalized: Normalized data
        @param xs: Data
        @param x_long: Data including further information such as realized returns, price_next, etc.
        @return: Dataframe containing the statistical evaluation
        """
        fig_dir = _resolve_figure_dir(fig_dir)

        #Add number of trees and imporant features as columns to output if only evaluating a tree model
        if len(self.model_types) == 1 and self.model_types[0] == 'rf':
            results = pd.DataFrame(columns=['Model', 'MSE', 'R^2-OS', 'Num_Trees', 'Important_Features',
                                            'Sharpe Ratio H-L', 'Clark-West'])
            total_num_trees = 0 #Initialize number of trees as zero
            # Initialize importance of features as zero
            totalImportance = (self.totalModels[0][0][0].feature_importance() -
                               self.totalModels[0][0][0].feature_importance())
        else:
            results = pd.DataFrame(columns=['Model', 'MSE', 'R^2-OS', 'Sharpe Ratio H-L', 'Clark-West'])

        #Get predictions for individual models to calculate individual statistics used for
        #example for Figure 4.2
        for i in range(self.N): #Iterate over rolling windows
            for j in range(self.amount): #Iterate over individual models
                allPreds = []
                for count, current_model in enumerate(self.model_names): #Iterate over model types
                    print(f'Current model evaluated: {current_model}')
                    if self.normalize[count]: #Use normalized data if applicable for option type
                        x = xs_normalized
                    else:
                        x = xs

                    if self.model_types[count] == 'rf': #Predict tree based models
                        model = self.totalModels[count][i][j]
                        y_pred = model.predict(x[i])
                        num_trees = model.num_trees() #get number of trees
                        # get feature importance in tree (determined by splits)
                        importance = model.feature_importance()
                        if len(self.model_types) == 1:
                            total_num_trees += num_trees
                            totalImportance += importance
                        #Keep 10 most important features to return in result
                        top_indices = np.argpartition(-importance, 10)[:10]
                        top_indices_sorted = top_indices[np.argsort(-importance[top_indices])]
                    else: #Predict neural network based models
                        #Create LargeDataset
                        ds = LargeDataset(x[i], self.important_columns, self.model_types[count])
                        loader = DataLoader(ds, batch_size=None)
                        y_preds = []
                        # Iterate through batches (main features, modulating features) in LargeDataset
                        for x_first, x_last in loader:
                            x_first, x_last = x_first.to('cuda'), x_last.to('cuda')
                            y_pre = self._forward_single_model(
                                self.totalModels[count][i][j], self.model_types[count], x_first, x_last
                            )
                            y_pre = y_pre.detach().cpu().numpy().flatten()
                            y_preds.append(y_pre)
                        y_pred = np.concatenate(y_preds) #concatenate results for batches

                    if self.normalize[count]: #unnormalize returns if model was trained with normalized data
                        normalizations = pd.read_excel('./normalization_params.xlsx', index_col=0)
                        y_pred = ((y_pred * normalizations.loc['std', 'Returns']) +
                                  normalizations.loc['mean', 'Returns'])
                        y_pred = y_pred.flatten()

                    allPreds.append(y_pred)

                #Average returns over model types using the respective weights (if we evaluate Ensemble model)
                averagedReturn = 0
                for k in range(len(self.model_names)):
                    averagedReturn += (self.model_weights[k] * allPreds[k])

                #Calculate MSE and R^2 using realized returns
                res = pd.DataFrame({'test': x_long[i]['Returns'].values, 'pred': averagedReturn})
                res['diff_sq'] = (res['test'] - res['pred']) ** 2
                res['test_sq'] = res['test'] ** 2
                mse = mean_squared_error(x_long[i]['Returns'], averagedReturn)
                r2_OS = 1 - (sum(res['diff_sq']) / sum(res['test_sq']))

                # add line to output if using a single tree-based method
                if len(self.model_types) == 1 and self.model_types[0] == 'rf':
                    results.loc[len(results)] = [f'Run {str(i)}/{str(j)}', mse, r2_OS, num_trees,
                                                 x[i].columns[top_indices_sorted], 'err', None]
                else: #add line to output
                    results.loc[len(results)] = [f'Run {str(i)}/{str(j)}', mse, r2_OS, 'err', None]

        #Create final row Ensemble
        #(actual model one would use and we generally mention throughout the thesis)
        x_long = pd.concat(x_long)
        y_pred = self.predict(xs_normalized, xs)
        y_test = x_long['Returns']

        #Calculate residuals and standardized residuals
        x_long['Prediction'] = y_pred
        x_long['Residuals'] = x_long['Returns'] - x_long['Prediction']
        x_long['std_Residuals'] = x_long['Residuals']/x_long['Residuals'].std(ddof=1)

        #Plot returns vs. standardized residuals (Figure 4.3)
        plt.figure()
        plt.plot(x_long['Returns'], x_long['std_Residuals'], 'o')
        plt.xlabel(f"Returns")
        plt.ylabel(f"Standardized Residuals")
        _save_fig(fig_dir, "residuals_vs_returns")

        #Plot prediction vs. standardized residuals (Figure 4.3)
        plt.figure()
        plt.plot(x_long['Prediction'], x_long['std_Residuals'], 'o')
        plt.xlabel(f"Prediction")
        plt.ylabel(f"Standardized Residuals")
        _save_fig(fig_dir, "residuals_vs_prediction")

        #Create QQ-plot (Figure 4.3)
        plt.figure()
        stats.probplot(x_long['std_Residuals'], dist="norm", plot=plt)
        plt.xlabel(f"Theoretical Quantiles")
        plt.ylabel(f"Empirical Quantiles")
        plt.title('')
        _save_fig(fig_dir, "qq_standardized_residuals")

        grouped = x_long.groupby(['loctimestamp', 'time']) #Group data by each timepoint
        cs = []
        for (loctimestamp, time), group in grouped: #iterate over timepoints
            n = len(group)
            if n == 0:
                result = 0
            else: #c for Clark-West test statistic (each timepoint represents one c)
                result = (1 / n) * ((group['Returns'] ** 2 - (group['Returns'] -
                                                              group['Prediction']) ** 2).sum())
            cs.append(result)

        #Perform calculation of Clark-West test statistic using 21*13 lags
        #(using diebold_mariano_method as it is the same calculation with model two set to zero prediction)
        cw, cw_p = CombinedModel.diebold_mariano_test(cs, 21*13, 'greater')

        #Calculate MSE and R^2 for ensemble using realized returns
        res = pd.DataFrame({'test': y_test.values, 'pred': y_pred})
        res['diff_sq'] = (res['test'] - res['pred']) ** 2
        res['test_sq'] = res['test'] ** 2
        mse = mean_squared_error(y_test, y_pred)
        r2_OS = 1 - (sum(res['diff_sq']) / sum(res['test_sq']))

        # add line to output if using a single tree-based method
        if len(self.model_types) == 1 and self.model_types[0] == 'rf':
            top_indices = np.argpartition(-totalImportance, 10)[:10]
            top_indices_sorted = top_indices[np.argsort(-totalImportance[top_indices])]
            results.loc[len(results)] = ['Ensemble', mse, r2_OS, total_num_trees,
                                         xs[0].columns[top_indices_sorted], 'err', cw_p]
        else: #add line to output
            results.loc[len(results)] = ['Ensemble', mse, r2_OS, 'err', cw_p]
        return results

    @staticmethod
    def diebold_mariano_test(d: list[float], h: int = 1, alternative: str = 'two-sided') \
            -> tuple[float, float]:
        """
        @brief Perform the Diebold-Mariano test or Clark-West test using Newey-West standard errors.
        @param d: d used in equation for Diebold Mariano test or c if using Clark-West test
        @param h: Maxlag
        @param alternative: Alternative to test
        @return: (test_statistic, p_value)
        """
        d_mean = np.mean(d) #mean of d

        #Fit constant model with Newey-West HAC standard errors
        d = d - d_mean  #demean for HAC
        X = np.ones(len(d))  #constant regressor
        model = sm.OLS(d, X)
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': h - 1}) #fit model

        dm_stat = d_mean / results.bse[0]  #test statistic

        #Two-sided or one-sided p-values
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
        elif alternative == 'greater':
            p_value = 1 - stats.norm.cdf(dm_stat)
        elif alternative == 'less':
            p_value = stats.norm.cdf(dm_stat)
        else:
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

        return dm_stat, p_value

    @staticmethod
    def process_group(group: pd.DataFrame, portf: int, long_short: bool, rho: float) -> pd.DataFrame:
        """
        @brief Function to group predictions into deciles/percentiles, etc. and calculate statistics
        @param group: Group of predictions (one timepoint)
        @param portf: How many portfolios to form (10 for deciles)
        @param long_short: Whether to calculate long-short portfolio or statistics for deciles/percentiles
        @param rho: Fraction of effective option spreads compared to quoted option spreads
        @return: Returns data for timepoint
        """
        group = group.sort_values(by='Prediction').copy()  #Sort by Prediction

        #Sort predictions into buckets. If group can't be spread to deciles/percentiles equally,
        #give the middle deciles/percentiles the extra contracts
        n = len(group)
        ideal_size = round(n / portf)
        buckets = [ideal_size] * portf
        extra = sum(buckets) - n
        middle_indices = range(1,portf-1)
        for i in middle_indices:
            if extra == 0:
                break
            buckets[i] -= np.sign(extra)
            extra -= np.sign(extra)
        if extra != 0:
            buckets[int(portf/2)] -= np.sign(extra)
        group['subgroup'] = np.concatenate([np.full(size, i) for i, size in enumerate(buckets)])

        if long_short:
            long_leg = group[group['subgroup'] == portf - 1].copy()
            short_leg = group[group['subgroup'] == 0].copy()
            if long_leg.empty or short_leg.empty:
                return pd.DataFrame({
                    'Prediction': [np.nan],
                    'Returns': [np.nan],
                    'Returns_optSpread': [np.nan],
                    'Returns_allSpread': [np.nan],
                    'Delta_change': [np.nan],
                    'Long_pos': [[]],
                    'Short_pos': [[]],
                    'Underlying_excess_return': [np.nan],
                    'valid_row': [False]
                })

            group = group[group['subgroup'].isin([0,portf-1])].copy() #only keep first and last portfolio
            group['Delta_change'] = group['delta_nex'] - group['delta'] #calculate change in delta

            # negate delta change for short portfolio
            group.loc[group['subgroup'] == 0, 'Delta_change'] *= -1
            # calculate absolute mean delta change over long-short portfolio
            nu = np.abs(np.mean(group['Delta_change']))

            # calculate realized costs for opening the option position
            group['tc_open_opt'] = (rho * group['optspread'] * group['price']) / 2
            # calculate realized costs for closing the option position
            group['tc_close_opt'] = (rho * group['optspread_nex'] * group['price_nex']) / 2
            # calculate realized costs for opening option and underlying position
            group['tc_open_all'] = (rho*group['optspread']*group['price'] +
                                    nu*group['undspread']*group['underlyingprice'])/2
            # calculate realized costs for closing option and underlying position
            group['tc_close_all'] = (rho*group['optspread_nex']*group['price_nex'] +
                                     nu*group['undspread_nex']*group['underlyingprice_nex'])/2

            #negate costs for options we short
            group.loc[group['subgroup'] == 0,
                ['tc_open_opt', 'tc_close_opt', 'tc_open_all', 'tc_close_all']] *= -1

            #Calculate returns using only option spread or both following formulas
            group['Returns_optSpread'] = group['Returns'] * np.abs(group['price']
                                        - group['delta'] * group['underlyingprice'])
            group['Returns_allSpread'] = ((group['Returns_optSpread'] - (group['riskfree']/(252*13)) *
                                        group['tc_open_all'] - group['tc_open_all'] - group['tc_close_all'])
                                        / (np.abs(group['price'] - group['delta'] * group['underlyingprice'])
                                        + group['tc_open_all']))
            group['Returns_optSpread'] = ((group['Returns_optSpread'] - (group['riskfree'] / (252 * 13))
                                        * group['tc_open_opt'] - group['tc_open_opt'] - group['tc_close_opt'])
                                        / (np.abs(group['price'] - group['delta'] * group['underlyingprice'])
                                        + group['tc_open_opt']))

            #negate values for shorting portfolio
            group.loc[group['subgroup'] == 0,
                ['Prediction', 'Returns', 'Returns_optSpread', 'Returns_allSpread']] *= -1

            #calculate excess return of underlying
            group['underlying_excess_return'] = ((group['underlyingprice_nex']/group['underlyingprice']) -
                                                 1 - (group['riskfree']/(252*13)))

            #return average prediction, returns, etc. for long-short portfolio.
            # Times two used for long and short position
            return pd.DataFrame({
                'Prediction': [2*group['Prediction'].mean()],
                'Returns': [2*group['Returns'].mean()],
                'Returns_optSpread': [2*group['Returns_optSpread'].mean()],
                'Returns_allSpread': [2*group['Returns_allSpread'].mean()],
                'Delta_change': [abs(group['Delta_change'].mean())],
                'Long_pos': [list(group[group['subgroup'] == portf-1]['uniqueIdent'])],
                'Short_pos': [list(group[group['subgroup'] == 0]['uniqueIdent'])],
                'Underlying_excess_return': [group['underlying_excess_return'].mean()],
                'valid_row': [True]
            })

        #return average values for each portfolio formed
        return group.groupby('subgroup').agg({
            'Prediction': 'mean',
            'Returns': 'mean',
            'optspread': 'mean',
            'delta': 'mean',
            'vega': 'mean',
            'theta': 'mean',
            'gamma': 'mean',
            'normalizedMoneyness': 'mean',
            'implVol': 'mean',
            'ask_size': 'mean',
            'bid_size': 'mean',
            'daystomaturity': 'mean',
            'putcall_P': 'mean',
        }).reset_index()

    @staticmethod
    def process_only_top_contenders(group: pd.DataFrame, rho: float) -> pd.DataFrame:
        """
        @brief This function calculates predicted and realized returns for a strategy,
            that shorts all negative predicted options and goes positive predicted ones long.
            (Selective Long-Short Portfolios)
        @param group: Group of predictions (one timepoint)
        @param rho: Fraction of effective spreads compared to quoted spreads
        @return: Returns data for timepoint
        """
        # only keep contracts with non zero predicted returns (after accounting for trading costs)
        group = group[np.abs(group['Prediction']) > 0]

        # calculate realized costs for opening option and underlying position
        group['tc_open_all'] = (rho * group['optspread'] * group['price'] + rho *
                                group['undspread'] * group['underlyingprice']) / 2
        # calculate realized costs for closing option and underlying position
        group['tc_close_all'] = (rho * group['optspread_nex'] * group['price_nex'] +
                                 rho * group['undspread_nex'] * group['underlyingprice_nex']) / 2

        #negate values for shorted options
        group.loc[group['Prediction'] < 0, ['tc_open_all', 'tc_close_all']] *= -1

        #Calculate returns using option and underlying spreads following formulas
        group['Returns_optSpread'] = group['Returns'] * np.abs(group['price'] -
                                    group['delta'] * group['underlyingprice'])
        group['Returns_allSpread'] = (group['Returns_optSpread'] - (group['riskfree'] /
                                    (252 * 13)) * group['tc_open_all'] - group['tc_open_all'] -
                                    group['tc_close_all']) / (np.abs(group['price'] -
                                    group['delta'] * group['underlyingprice']) + group['tc_open_all'])

        #negate values for shorted options
        group.loc[group['Prediction'] < 0, ['Prediction', 'Returns_allSpread']] *= -1

        #return average prediction and realized returns
        return pd.DataFrame({
            'Prediction': [2 * group['Prediction'].mean()],
            'Returns': [2 * group['Returns_allSpread'].mean()],
        })

    @staticmethod
    def compute_turnover(df: pd.DataFrame) -> pd.DataFrame:
        """
        @brief Function to calculate the turnover in portfolios between two timepoints
        @param df: Portfolio
        @return: add Turnover column to portfolio
        """
        turnovers = [0]  #First row has no prior row
        for i in range(1, len(df)): #iterate over rows (timepoints)
            #Identify previous and current long and short positions
            prev_longs = set(df.loc[i - 1, 'Long_pos'])
            prev_shorts = set(df.loc[i - 1, 'Short_pos'])
            curr_longs = set(df.loc[i, 'Long_pos'])
            curr_shorts = set(df.loc[i, 'Short_pos'])

            #Calculate changed positions
            changed_long = prev_longs.symmetric_difference(curr_longs)
            changed_short = prev_shorts.symmetric_difference(curr_shorts)

            #Calculate turnover in long and short portfolio
            #(2 comes from the symmetric_difference method that counts changes twice)
            turnover_long = len(changed_long) / (2*len(prev_longs) if len(prev_longs) != 0 else 1)
            turnover_short = len(changed_short) / (2*len(prev_shorts) if len(prev_longs) != 0 else 1)

            #Calculate mean turnover
            turnovers.append((turnover_long + turnover_short)/2)

        df['turnover'] = turnovers
        return df

    def sharpe_top_contenders(self, x: pd.DataFrame, y_pred: pd.Series, rho: float) \
            -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        @brief This function calculates sharpe ratios for a strategy, that shorts all negative
            predicted options and goes positive predicted ones long. (Selective Long-Short Portfolios)
        @param x: Data containing realized returns
        @param y_pred: Data containing predictions
        @param rho: Fraction of effective spreads compared to quoted spreads
        @return: Both aggreagated results and results for each timepoint
        """
        x['Prediction'] = y_pred #add prediction to x

        #Estimate trading costs and calculate prediction when accounting for those costs.
        #A strategy with previous positive returns, that results in negative predicted returns after
        # accounting for trading costs, will be set to zero, so that we don't short this strategy,
        # and vice versa.
        x['tc_all'] = (rho * x['optspread'] * x['price'] + rho*x['undspread']*x['underlyingprice']) / 2
        x['Prediction_Less_Denominator'] = x['Prediction'] * np.abs(x['price'] -
                                                                    x['delta'] * x['underlyingprice'])
        x['Prediction'] = np.where(np.sign(x['Prediction']) > 0, np.maximum(0,
                        (x['Prediction_Less_Denominator'] - (2+(x['riskfree'] / (252 * 13)))*x['tc_all'])/
                        (np.abs(x['price'] - x['delta'] * x['underlyingprice'])+x['tc_all'])),
                        np.where(np.sign(x['Prediction']) < 0, np.minimum(0,
                        (x['Prediction_Less_Denominator'] + (2+(x['riskfree'] / (252 * 13)))*
                         x['tc_all'])/(np.abs(x['price'] - x['delta'] * x['underlyingprice'])-x['tc_all'])), 0))

        len_used = len(x.loc[np.abs(x['Prediction']) > 0])

        #group data by timepoint and perform return calculation for strategy
        new_rows = x.groupby(['loctimestamp', 'time'], group_keys=False).apply(
            lambda y: CombinedModel.process_only_top_contenders(y, rho)).reset_index()
        new_rows['subgroup'] = f'{rho} turnover'
        result = pd.concat([new_rows], ignore_index=True) #concat data for timepoints

        #Aggregate means for timepoints
        result_portfolios = pd.DataFrame({
            'subgroup': [f'H-L {rho} turnover'],
            'pred_mean': [result['Prediction'].fillna(0).mean()],
            'return_mean': [result['Returns'].fillna(0).mean()],
            'return_std': [result['Returns'].fillna(0).std()],
            'len_used': [len_used], #Amount of contracts actually traded
            'len_possible': [len(x['Prediction'])], #Total amount of contracts
            'percentage_used': [len_used/len(x['Prediction'])], #Percentage of traded contracts
        })

        #Calculate sharpe ratio
        # Sharpe Ratio for 30 min interval
        result_portfolios['sharpe'] = result_portfolios['return_mean'] / result_portfolios['return_std']
        # Annualized returns
        result_portfolios['return_mean_yearly'] = ((1 + result_portfolios['return_mean']) ** (252 * 13)) - 1
        # Annualized volatility
        result_portfolios['return_std_yearly'] = result_portfolios['return_std'] * np.sqrt(252 * 13)
        # Annualized Sharpe Ratio
        result_portfolios['SR_yearly'] = result_portfolios['sharpe'] * np.sqrt(252 * 13)

        return result_portfolios, result #Return both aggreagated results and results for each timepoint

    def sharpe(self, x: pd.DataFrame, y_pred: pd.Series, opt_spread: bool, und_spread: bool,
               rho: float, portf: int, graph: bool = False, fig_dir=None,
               figure_prefix: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        @brief This function calculates sharpe ratios for a long-short strategy based on deciles/percentiles
        @param x: Data containing realized returns
        @param y_pred: Data containing predictions
        @param opt_spread: Whether to include option spreads
        @param und_spread: Whether to include underlying spreads
            (only works with option spreads enabled as well)
        @param rho: Fraction of effective option spreads compared to quoted option spreads
        @param portf: How many portfolios to form (10 for deciles)
        @param graph: Whether to plot strategy returns vs. market returns (Figure 4.8)
        @return: Both aggreagated results and results for each timepoint
        """
        fig_dir = _resolve_figure_dir(fig_dir) if graph and fig_dir is not None else fig_dir
        x = x.copy()
        x['Prediction'] = y_pred #add prediction to x
        subgroup_label = f'H-L {portf} portfolios, {rho} turnover'

        def _empty_long_short_result() -> pd.DataFrame:
            empty_row = {
                'loctimestamp': pd.NaT,
                'time': np.nan,
                'Prediction': np.nan,
                'Returns': np.nan,
                'Returns_optSpread': np.nan,
                'Returns_allSpread': np.nan,
                'Delta_change': np.nan,
                'Long_pos': [],
                'Short_pos': [],
                'Underlying_excess_return': np.nan,
                'valid_row': False,
                'subgroup': subgroup_label,
            }
            if opt_spread:
                empty_row['turnover'] = np.nan
            return pd.DataFrame([empty_row])

        # Assume fraction of effective underlying spreads compared to quoted underlying spreads
        # (only for prediction), calculated extra for realized returns
        nu = 0.003

        #Estimate trading costs and calculate prediction when accounting for those costs.
        #A strategy with previous positive returns, that results in negative predicted returns after
        # accounting for trading costs, will be set to zero, so that we don't short this strategy,
        # and vice versa.
        x['tc_opt'] = (rho * x['optspread'] * x['price']) / 2
        x['tc_all'] = (rho * x['optspread'] * x['price'] + nu*x['undspread']*x['underlyingprice']) / 2
        x['Prediction_Less_Denominator'] = x['Prediction'] * np.abs(x['price'] - x['delta'] * x['underlyingprice'])
        if opt_spread and not und_spread:
            x['Prediction'] = np.where(np.sign(x['Prediction']) > 0, np.maximum(0,
                            (x['Prediction_Less_Denominator'] - (2+(x['riskfree'] / (252 * 13)))*
                            x['tc_opt'])/(np.abs(x['price'] - x['delta'] * x['underlyingprice'])+
                            x['tc_opt'])), np.where(np.sign(x['Prediction']) < 0, np.minimum(0,
                            (x['Prediction_Less_Denominator'] + (2+(x['riskfree'] / (252 * 13)))*
                            x['tc_opt'])/(np.abs(x['price'] - x['delta'] * x['underlyingprice'])-
                            x['tc_opt'])), 0))
        if opt_spread and und_spread:
            x['Prediction'] = np.where(np.sign(x['Prediction']) > 0, np.maximum(0,
                            (x['Prediction_Less_Denominator'] - (2+(x['riskfree'] / (252 * 13)))*
                            x['tc_all'])/(np.abs(x['price'] - x['delta'] * x['underlyingprice'])+
                            x['tc_all'])), np.where(np.sign(x['Prediction']) < 0, np.minimum(0,
                            (x['Prediction_Less_Denominator'] + (2+(x['riskfree'] / (252 * 13)))*
                            x['tc_all'])/(np.abs(x['price'] - x['delta'] * x['underlyingprice'])-
                            x['tc_all'])), 0))
        if not opt_spread and und_spread:
            raise ValueError('Combination of only Underlying Spread not implemented.')

        #If we don't use trading costs, also return the stats for individual portfolios
        # (Sheet Ensemble in performance.xlsx)
        if not opt_spread:
            #group data by timepoint and perform return calculation for individual portfolios
            result = x.groupby(['loctimestamp', 'time'], group_keys=False).apply(
                lambda y: CombinedModel.process_group(y, portf, False, rho)).reset_index()

        #group data by timepoint and perform return calculation for long-short strategy
        new_rows = x.groupby(['loctimestamp', 'time'], group_keys=False).apply(
            lambda y: CombinedModel.process_group(y, portf, True, rho)).reset_index()
        new_rows = new_rows.replace([np.inf, -np.inf], np.nan)
        new_rows['subgroup'] = subgroup_label
        new_rows_all = new_rows.copy()
        if 'valid_row' in new_rows.columns:
            new_rows_valid = new_rows[new_rows['valid_row'] == True].copy()
        else:
            new_rows_valid = new_rows.copy()

        #Append the new rows to the dataframe
        # report only return statistics and delta change / turnover for long-short portfolios
        # with spreads (Sheets Ensemble optspreads and Ensemble opt-undspreads in performance.xlsx)
        if opt_spread:
            new_rows_all['Returns'] = new_rows_all['Returns_optSpread'] #use returns with option spreads
            new_rows_valid['Returns'] = new_rows_valid['Returns_optSpread']
            if und_spread: #use returns with also underlying spreads
                new_rows_all['Returns'] = new_rows_all['Returns_allSpread']
                new_rows_valid['Returns'] = new_rows_valid['Returns_allSpread']
            valid_result = pd.concat([new_rows_valid], ignore_index=True)
            if valid_result.empty:
                result = _empty_long_short_result()
            else:
                result = CombinedModel.compute_turnover(valid_result.copy()) #compute turnover in long-short portfolio
            turnover_result = (CombinedModel.compute_turnover(valid_result.copy())
                               if not valid_result.empty else None)
            #Aggregate means for each subgroup (timepoint)
            result_portfolios = pd.DataFrame({
                'subgroup': [subgroup_label],
                'pred_mean': [valid_result['Prediction'].mean()],
                'return_mean': [valid_result['Returns'].mean()],
                'return_std': [valid_result['Returns'].std()],
                'delta_change_mean': [valid_result['Delta_change'].mean()],
                'delta_change_std': [valid_result['Delta_change'].std()],
                'turnover_mean': [turnover_result['turnover'].iloc[1:].mean() if turnover_result is not None else np.nan],
                'turnover_std': [turnover_result['turnover'].iloc[1:].std() if turnover_result is not None else np.nan],
            })
        # use data for individual portfolios and long-short portfolio and also give mean feature
        # values for portfolios (Sheet Ensemble in performance.xlsx and Table 4.4)
        else:
            result_summary = pd.concat([result, new_rows_valid], ignore_index=True)
            result = result_summary.copy()
            if new_rows_valid.empty:
                result = pd.concat([result, _empty_long_short_result()], ignore_index=True)
            #Aggregate means for each subgroup (timepoint)
            result_portfolios = result_summary.groupby('subgroup').agg({
                'Prediction': 'mean',
                'Returns': ['mean', 'std'],
                'optspread': 'mean',
                'delta': 'mean',
                'vega': 'mean',
                'theta': 'mean',
                'gamma': 'mean',
                'normalizedMoneyness': 'mean',
                'implVol': 'mean',
                'ask_size': 'mean',
                'bid_size': 'mean',
                'daystomaturity': 'mean',
                'putcall_P': 'mean',
            }).reset_index()
            result_portfolios.columns = ['subgroup', 'pred_mean', 'return_mean', 'return_std',
                                         'optspread_mean', 'delta_mean', 'vega_mean', 'theta_mean',
                                         'gamma_mean', 'normalizedMoneyness_mean', 'implVol_mean',
                                         'ask_size_mean', 'bid_size_mean', 'daystomaturity_mean',
                                         'portion_put_mean']
            if not result_portfolios['subgroup'].eq(subgroup_label).any():
                result_portfolios = pd.concat([result_portfolios, pd.DataFrame([{
                    'subgroup': subgroup_label,
                    'pred_mean': np.nan,
                    'return_mean': np.nan,
                    'return_std': np.nan,
                    'optspread_mean': np.nan,
                    'delta_mean': np.nan,
                    'vega_mean': np.nan,
                    'theta_mean': np.nan,
                    'gamma_mean': np.nan,
                    'normalizedMoneyness_mean': np.nan,
                    'implVol_mean': np.nan,
                    'ask_size_mean': np.nan,
                    'bid_size_mean': np.nan,
                    'daystomaturity_mean': np.nan,
                    'portion_put_mean': np.nan,
                }])], ignore_index=True)

        #Perform t-test for significance against market returns
        capm_df = new_rows_valid[['Returns', 'Underlying_excess_return']].copy()
        capm_df = capm_df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(capm_df) >= 3 and capm_df['Underlying_excess_return'].nunique() >= 2:
            strat = capm_df['Returns']
            market = capm_df['Underlying_excess_return']
            t, p = stats.ttest_1samp((strat - market).tolist(), popmean=0)
            print(f'Portfolios {portf}, Rho {rho}: Strategy vs S&P 500 t = {str(t)}, p = {str(p)}')

            #Perform CAPM regression
            regr = sm.OLS(strat, sm.add_constant(market)).fit()
            params = regr.params
            alpha, alpha_pvalue = params[0], regr.pvalues[0]
            beta, beta_pvalue = params[1], regr.pvalues[1]
            x_axis = np.linspace(-0.01, 0.01, 1000)
            y_axis = alpha + x_axis * beta

            if graph: #If wished, plot CAPM regression (Figure 4.8)
                plt.figure()
                plt.plot(market, strat, 'o')
                plt.plot(x_axis, y_axis, label="Regression Result", linewidth=5)
                plt.plot(x_axis, x_axis, label="Return S&P 500", linewidth=5)
                plt.legend()
                plt.xlabel(f"Return S&P 500")
                plt.ylabel(f"Return Long-Short")
                figure_name = f"capm_regression_portf_{portf}_rho_{rho}"
                if figure_prefix:
                    figure_name = f"{figure_prefix}_{figure_name}"
                _save_fig(_resolve_figure_dir(fig_dir), figure_name)
        else:
            t = np.nan
            p = np.nan
            alpha = np.nan
            alpha_pvalue = np.nan
            beta = np.nan
            beta_pvalue = np.nan
            print(f'Warning: skipped CAPM regression for portf={portf}, rho={rho} because valid rows are insufficient after filtering.')

        #Calculate Sharpe Ratios per portfolio and add to df
        # Sharpe Ratio for 30 min interval
        result_portfolios['sharpe'] = result_portfolios['return_mean'] / result_portfolios['return_std']
        # Annualized returns
        result_portfolios['return_mean_yearly'] = ((1 + result_portfolios['return_mean']) ** (252 * 13)) - 1
        # Annualized volatility
        result_portfolios['return_std_yearly'] = result_portfolios['return_std'] * np.sqrt(252 * 13)
        # Annualized Sharpe Ratio
        result_portfolios['SR_yearly'] = result_portfolios['sharpe'] * np.sqrt(252 * 13)

        #Add CAPM alpha, beta and Ret vs. S&P test statistics and p-values to df for long-short portfolio
        result_portfolios['CAPM [alpha, pvalue]'] = None
        result_portfolios['CAPM [beta, pvalue]'] = None
        result_portfolios['Ret vs. S&P [t,p]'] = None
        long_short_indices = result_portfolios.index[result_portfolios['subgroup'] == subgroup_label]
        if len(long_short_indices) > 0:
            long_short_index = long_short_indices[0]
            result_portfolios.at[long_short_index, 'CAPM [alpha, pvalue]'] = [alpha, alpha_pvalue]
            result_portfolios.at[long_short_index, 'CAPM [beta, pvalue]'] = [beta, beta_pvalue]
            result_portfolios.at[long_short_index, 'Ret vs. S&P [t,p]'] = [t, p]

        return result_portfolios, result #Return both aggreagated results and results for each timepoint

    def robustness(self,
                   x_cleans_normalized: list[pd.DataFrame],
                   x_cleans: list[pd.DataFrame],
                   out_dir: str = "robustness",
                   save_table: bool = True,
                   save_fig: bool = True,
                   seed: int = 0) -> None:
        """
        @brief This function performs the noise robustness test as described in Section 4.6
        @param x_cleans_normalized: Features for prediction normalized
        @param x_cleans: Features for prediction
        """
        from pathlib import Path
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        rows = []

        y_pred_clean = self.predict(x_cleans_normalized, x_cleans) #calculate predictions for data
        denom = np.maximum(np.abs(y_pred_clean), 1e-12)
        robustnessScore = []
        mean_drifts = []
        noise_levels = [0.001, 0.0025, 0.005, 0.01]
        for noise_level in noise_levels: #iterate through noise levels
            y_preds = []
            for mc_index in range(25): #iterate through 25 Monte Carlo simulation runs
                print(f'Robustness for noise level {noise_level}, Monte Carlo run {mc_index}')
                #add gaussian noise with noise level to normalized data
                X_normalized_noisy = [
                    df.assign(**{
                        col: df[col] + rng.normal(0, noise_level, size=len(df)) * df[col].values
                        for col in df.columns
                    })
                    for df in x_cleans_normalized
                ]
                #add gaussian noise with noise level to data
                X_noisy = [
                    df.assign(**{
                        col: df[col] + rng.normal(0, noise_level, size=len(df)) * df[col].values
                        for col in df.columns
                    })
                    for df in x_cleans
                ]
                #Calculate prediction using noisy data
                y_pred_run = self.predict(X_normalized_noisy, X_noisy)
                y_preds.append(y_pred_run)
                drift_run = np.mean(np.abs(y_pred_run - y_pred_clean) / denom)
                rows.append({"noise": noise_level, "mc": mc_index, "drift_run": float(drift_run)})

            y_preds = np.array(y_preds)
            #Calculate prediction variance per Monte Carlo run
            std_dev_per_sample = np.std(y_preds, axis=0) / denom
            #Calculate prediction drift per Monte Carlo run
            bias_per_sample = np.abs(np.mean(y_preds, axis=0) - y_pred_clean) / denom
            #Average over Monte Carlo runs
            robustnessScore.append(np.mean(std_dev_per_sample))
            mean_drifts.append(np.mean(bias_per_sample))

        #Plot noise robustness (Figure 4.9)
        plt.plot(noise_levels, mean_drifts, label='Prediction Drift (Bias)', color="steelblue")
        plt.plot(noise_levels, robustnessScore, label='Prediction Std. Dev. (Variance)',
                 color="darkorange")
        plt.xlabel("Relative Noise Level")
        plt.ylabel("Error Magnitude")
        plt.legend()
        if save_fig:
            _save_fig(out_dir, "robustness", dpi=200)
        else:
            plt.close(plt.gcf())

        runs_df = pd.DataFrame(rows)
        if save_table:
            runs_df.to_parquet(out_dir/"robustness_runs.parquet", index=False)
            summary = (runs_df.groupby("noise")["drift_run"]
                              .agg(["mean", "std", "min", "max", "count"])
                              .reset_index())
            summary["drift_bias"] = [float(x) for x in mean_drifts]
            summary["variance_score"] = [float(x) for x in robustnessScore]
            summary.to_excel(out_dir/"robustness_summary.xlsx", index=False)

        print(f"[robustness] wrote: {out_dir/'robustness_runs.parquet'}")
        print(f"[robustness] wrote: {out_dir/'robustness_summary.xlsx'}")
        print(f"[robustness] wrote: {out_dir/'robustness.png'}")

    def performancePortfolios(self, x_cleans_normalized: list[pd.DataFrame],
                              x_cleans: list[pd.DataFrame], xs: list[pd.DataFrame],
                              basic_only: bool = False, output_dir=None, fig_dir=None) -> None:
        """
        @brief Main function to initiate creation of performance.xlsx and selective_strategy.xlsx
            Calculates R^2 and Sharpe Ratios for multiple long-short or selective strategies
        @param x_cleans_normalized: Normalized features
        @param x_cleans: Features
        @param xs: Data including further information such as realized returns, price_next, etc.
        @param basic_only: Whether to only create the base performance.xlsx output
            with Models and Ensemble sheets
        """

        output_dir = _resolve_output_dir(output_dir)
        fig_dir = _resolve_figure_dir(fig_dir, output_dir=output_dir)

        #Perform statistical evaluation of models
        evaluationModels = self.evaluate(x_cleans_normalized, x_cleans, xs, fig_dir=fig_dir)

        for j in range(self.N): #Iterate over rolling windows
            for i in range(self.amount): #Iterate over individual models per rolling window
                allPreds = []
                for count, current_model in enumerate(self.model_names): #iterate over model types
                    if self.normalize[count]: #use normalized data depending on model type
                        x_used = x_cleans_normalized
                    else:
                        x_used = x_cleans
                    x = xs[j] #take correct rolling window
                    if self.model_types[count] == 'rf': #predict for tree-based models
                        y_pred = self.totalModels[count][j][i].predict(x_used[j])
                    else: #predict for neural network based models
                        #create large dataset
                        ds = LargeDataset(x_used[j], self.important_columns, self.model_types[count])
                        loader = DataLoader(ds, batch_size=None)
                        y_preds = []
                        for x_first, x_last in loader: #iterate over batches
                            x_first, x_last = x_first.to('cuda'), x_last.to('cuda')
                            y_pred = self._forward_single_model(
                                self.totalModels[count][j][i], self.model_types[count], x_first, x_last
                            )
                            y_pred = y_pred.detach().cpu().numpy()
                            y_preds.append(y_pred)
                        y_pred = np.concatenate(y_preds) #concatenate results for batches
                    if self.normalize[count]: #unnormalize returns if model was trained with normalized data
                        normalizations = pd.read_excel('./normalization_params.xlsx', index_col=0)
                        y_pred = ((y_pred * normalizations.loc['std', 'Returns']) +
                                  normalizations.loc['mean', 'Returns'])
                        y_pred = y_pred.flatten()
                    allPreds.append(y_pred)

                #Average returns over model types using the respective weights (if we evaluate Ensemble model)
                averagedReturn = 0
                for k in range(len(self.model_names)):
                    averagedReturn += (self.model_weights[k] * allPreds[k])

                #Calculate Sharpe Ratio for each run for long-short strategy using no trading costs
                # and decile portfolios (Column Sharpe Ratio in Sheet Models in performance.xlsx)
                result_portfolios, result = self.sharpe(x, averagedReturn, False,
                                                        False, 0,  10)
                evaluationModels.iloc[j*self.amount + i, -2] = result_portfolios['SR_yearly'][10]

        #perform prediction for all runs combined
        y_pred = self.predict(x_cleans_normalized, x_cleans)
        x = pd.concat(xs)
        cross_section_sizes = x.groupby(['loctimestamp', 'time']).size()
        min_cs = cross_section_sizes.min()
        median_cs = cross_section_sizes.median()
        print(f'Cross-section stats: min={min_cs}, median={median_cs}, requested max portfolios=1000')
        print('If requested portfolios exceed available contracts in a timepoint, that timepoint will be skipped instead of forcing empty long-short buckets.')

        #Calculate Sharpe Ratio for long-short strategy using no trading costs and decile portfolios
        # and also plot CAPM regression (use also for strategy plotting later)
        result_portfolios, result = self.sharpe(x, y_pred, False, False, 0,
                                                10, not basic_only, fig_dir=fig_dir)
        # Column Sharpe Ratio, row Ensemble in performance.xlsx
        evaluationModels.iloc[self.N*self.amount, -2] = result_portfolios['SR_yearly'][10]

        if basic_only:
            writer = pd.ExcelWriter(output_dir / 'performance.xlsx', engine='xlsxwriter')
            evaluationModels.to_excel(writer, sheet_name='Models')
            result_portfolios.to_excel(writer, sheet_name='Ensemble')
            writer.close()
            return

        writer = pd.ExcelWriter(output_dir / 'selective_strategy.xlsx', engine='xlsxwriter')
        #Perform strategy calculation for only predicted non-zero returns (selective strategy)
        # using different fractions of effective to quoted trading costs
        for rho in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]: #fractions of quoted spreads to pay
            resultPort, result = self.sharpe_top_contenders(x, y_pred, rho) #perform strategy evaluation
            resultPort.to_excel(writer, sheet_name=f'result_{rho}') #aggregated results
            result.to_excel(writer, sheet_name=f'trading_times_{rho}') #results for each trading interval
        writer.close()

        #Calculate Sharpe Ratio for long-short strategy using no trading costs and
        #percentile portfolios (use for strategy plotting later)
        portfolios, result2 = self.sharpe(x, y_pred, False, False, 0, 100)

        result_portfolios_spread1 = None
        result_portfolios_spread2 = None
        total3dspread1 = []
        total3dspread2 = []
        for turnover in [0.05, 0.04, 0.03, 0.02, 0.01, 0]: #Iterate over different turnover assumptions
            three_d_spread1 = []
            three_d_spread2 = []


            # Perform calculation of Sharpe Ratios for multiple combinations of option spreads /
            # both spreads, tunover level and amount of portfolios. This will be put into Sheets
            # Ensemble optspreads and Ensemble opt-undspreads in performance.xlsx


            result_portfolios_spread, result_spread = self.sharpe(x, y_pred, True,
                                                                  False, turnover, 3)
            result_portfolios_spread1 = pd.concat([result_portfolios_spread1, result_portfolios_spread])
            three_d_spread1.append(result_portfolios_spread['SR_yearly'][0])
            result_portfolios_spread, result_spread = self.sharpe(x, y_pred, True,
                                                                  False, turnover, 4)
            result_portfolios_spread1 = pd.concat([result_portfolios_spread1, result_portfolios_spread])
            three_d_spread1.append(result_portfolios_spread['SR_yearly'][0])
            result_portfolios_spread, result_spread = self.sharpe(x, y_pred, True,
                                                                  False, turnover, 5)
            result_portfolios_spread1 = pd.concat([result_portfolios_spread1, result_portfolios_spread])
            three_d_spread1.append(result_portfolios_spread['SR_yearly'][0])

            result_portfolios_spread, result_spread = self.sharpe(x, y_pred, True,
                                                                  True, turnover, 3)
            result_portfolios_spread2 = pd.concat([result_portfolios_spread2, result_portfolios_spread])
            three_d_spread2.append(result_portfolios_spread['SR_yearly'][0])
            result_portfolios_spread, result_spread = self.sharpe(x, y_pred, True,
                                                                  True, turnover, 4)
            result_portfolios_spread2 = pd.concat([result_portfolios_spread2, result_portfolios_spread])
            three_d_spread2.append(result_portfolios_spread['SR_yearly'][0])
            result_portfolios_spread, result_spread = self.sharpe(x, y_pred, True,
                                                                  True, turnover, 5)
            result_portfolios_spread2 = pd.concat([result_portfolios_spread2, result_portfolios_spread])
            three_d_spread2.append(result_portfolios_spread['SR_yearly'][0])

            result_portfolios_spread, result_spread = self.sharpe(x, y_pred, True,
                                                                  False, turnover, 10)
            result_portfolios_spread1 = pd.concat([result_portfolios_spread1, result_portfolios_spread])
            three_d_spread1.append(result_portfolios_spread['SR_yearly'][0])
            result_portfolios_spread, result_spread = self.sharpe(x, y_pred, True,
                                                                  True, turnover, 10)
            result_portfolios_spread2 = pd.concat([result_portfolios_spread2, result_portfolios_spread])
            three_d_spread2.append(result_portfolios_spread['SR_yearly'][0])

            result_portfolios_spread, result_spread = self.sharpe(x, y_pred, True,
                                                                  False, turnover, 20)
            result_portfolios_spread1 = pd.concat([result_portfolios_spread1, result_portfolios_spread])
            three_d_spread1.append(result_portfolios_spread['SR_yearly'][0])
            result_portfolios_spread, result_spread = self.sharpe(x, y_pred, True,
                                                                  False, turnover, 40)
            result_portfolios_spread1 = pd.concat([result_portfolios_spread1, result_portfolios_spread])
            three_d_spread1.append(result_portfolios_spread['SR_yearly'][0])
            result_portfolios_spread, result_spread1 = self.sharpe(x, y_pred, True,
                                                                   False, turnover, 100)
            if turnover == 0.01: #save for plotting later
                extra1 = result_spread1
            result_portfolios_spread1 = pd.concat([result_portfolios_spread1, result_portfolios_spread])
            three_d_spread1.append(result_portfolios_spread['SR_yearly'][0])

            result_portfolios_spread, result_spread = self.sharpe(x, y_pred, True,
                                                                  False, turnover, 200)
            result_portfolios_spread1 = pd.concat([result_portfolios_spread1, result_portfolios_spread])
            three_d_spread1.append(result_portfolios_spread['SR_yearly'][0])
            result_portfolios_spread, result_spread = self.sharpe(x, y_pred, True,
                                                                  False, turnover, 500)
            result_portfolios_spread1 = pd.concat([result_portfolios_spread1, result_portfolios_spread])
            three_d_spread1.append(result_portfolios_spread['SR_yearly'][0])
            result_portfolios_spread, result_spread = self.sharpe(x, y_pred, True,
                                                                  False, turnover, 1000)
            result_portfolios_spread1 = pd.concat([result_portfolios_spread1, result_portfolios_spread])

            result_portfolios_spread, result_spread = self.sharpe(x, y_pred, True,
                                                                  True, turnover, 20)
            result_portfolios_spread2 = pd.concat([result_portfolios_spread2, result_portfolios_spread])
            three_d_spread2.append(result_portfolios_spread['SR_yearly'][0])
            result_portfolios_spread, result_spread = self.sharpe(x, y_pred, True,
                                                                  True, turnover, 40)
            result_portfolios_spread2 = pd.concat([result_portfolios_spread2, result_portfolios_spread])
            three_d_spread2.append(result_portfolios_spread['SR_yearly'][0])
            result_portfolios_spread, result_spread2 = self.sharpe(x, y_pred, True,
                                                                   True, turnover, 100)
            result_portfolios_spread2 = pd.concat([result_portfolios_spread2, result_portfolios_spread])
            if turnover == 0.01: #save for plotting later
                extra2 = result_spread2
            three_d_spread2.append(result_portfolios_spread['SR_yearly'][0])

            result_portfolios_spread, result_spread = self.sharpe(x, y_pred, True,
                                                                  True, turnover, 200)
            result_portfolios_spread2 = pd.concat([result_portfolios_spread2, result_portfolios_spread])
            three_d_spread2.append(result_portfolios_spread['SR_yearly'][0])
            result_portfolios_spread, result_spread = self.sharpe(x, y_pred, True,
                                                                  True, turnover, 500)
            result_portfolios_spread2 = pd.concat([result_portfolios_spread2, result_portfolios_spread])
            three_d_spread2.append(result_portfolios_spread['SR_yearly'][0])
            result_portfolios_spread, result_spread = self.sharpe(x, y_pred, True,
                                                                  True, turnover, 1000)
            result_portfolios_spread2 = pd.concat([result_portfolios_spread2, result_portfolios_spread])

            total3dspread1.append(three_d_spread1)
            total3dspread2.append(three_d_spread2)

        #Create plot for cumulative return of decile strategy without trading costs or
        # turnover over testing sample
        hlPortfolio1 = result[result['subgroup'] == 'H-L 10 portfolios, 0 turnover']
        hlPortfolio1 = hlPortfolio1.copy()
        if hlPortfolio1.empty:
            print('Warning: skipped cumulative_return_deciles because no decile strategy returns were available.')
        else:
            hlPortfolio1['cumRet'] = (1 + hlPortfolio1['Returns']).cumprod()
            hlPortfolio1 = hlPortfolio1.reset_index()
            plt.figure()
            plt.plot(range(len(hlPortfolio1['cumRet'])), hlPortfolio1['cumRet'],
                     label='H-L Deciles')
            plt.xlabel('Time')
            plt.ylabel('Cumulative Return')
            plt.legend()
            _save_fig(fig_dir, "cumulative_return_deciles")

        #Create plot for cumulative returns of percentile strategies using no trading costs,
        # only option costs and both costs for 1% turnover (Figure B.1)
        hlPortfolio = result2[result2['subgroup'] == 'H-L 100 portfolios, 0 turnover'].copy()
        hlPortfolio_spread1 = extra1[
            extra1['subgroup'] == 'H-L 100 portfolios, 0.01 turnover'
        ].copy()
        hlPortfolio_spread2 = extra2[
            extra2['subgroup'] == 'H-L 100 portfolios, 0.01 turnover'
        ].copy()
        if hlPortfolio.empty or hlPortfolio_spread1.empty or hlPortfolio_spread2.empty:
            print('Warning: skipped cumulative_return_percentiles_spreads because one or more percentile strategy series were empty.')
        else:
            hlPortfolio['cumRet'] = (1 + hlPortfolio['Returns']).cumprod()
            hlPortfolio_spread1['cumRet'] = (1 + hlPortfolio_spread1['Returns']).cumprod()
            hlPortfolio_spread2['cumRet'] = (1 + hlPortfolio_spread2['Returns']).cumprod()
            plt.figure()
            plt.plot(range(len(hlPortfolio['cumRet'])), hlPortfolio['cumRet'],
                     label='H-L Percentiles')
            plt.plot(range(len(hlPortfolio_spread1['cumRet'])), hlPortfolio_spread1['cumRet'],
                     label='H-L Percentiles Option Spreads')
            plt.plot(range(len(hlPortfolio_spread2['cumRet'])), hlPortfolio_spread2['cumRet'],
                     label='H-L Percentiles All Spreads')
            plt.xlabel('Time')
            plt.ylabel('Cumulative Return')
            plt.legend()
            _save_fig(fig_dir, "cumulative_return_percentiles_spreads")

        #Create density plot for predicted and realized returns across dataset (Figure 4.3)
        if sns is None:
            print('Warning: skipped return_density_predicted_vs_realized because seaborn is not available.')
        else:
            plt.figure()
            sns.kdeplot(y_pred, label='Predicted Returns', fill=True)
            sns.kdeplot(x['Returns'], label='Realized Returns', fill=True)
            plt.xlabel('Return')
            plt.ylabel('Density')
            plt.xlim(-0.0025, 0.0025)
            plt.legend()
            _save_fig(fig_dir, "return_density_predicted_vs_realized")

        #adjust plotting params
        plt.rcParams.update({'font.size': 10})

        #3d plot of Sharpe Ratios for different strategies using option spreads
        data = np.array(total3dspread1)
        M, D = np.meshgrid([3,4,5,10, 20, 40, 100, 200, 500], [0.05, 0.04, 0.03, 0.02, 0.01, 0])
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(D, M, data, cmap='viridis')
        ax.set_xlabel('Effective Option Spread')
        ax.set_ylabel('Portfolio Buckets')
        ax.set_zlabel('Sharpe Ratio')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        _save_fig(fig_dir, "sharpe_surface_option_spreads")

        #3d plot of Sharpe Ratios for different strategies using option and underlying spreads (Figure B.2)
        data = np.array(total3dspread2)
        M, D = np.meshgrid([3,4,5,10, 20, 40, 100, 200, 500], [0.05, 0.04, 0.03, 0.02, 0.01, 0])
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(D, M, data, cmap='viridis')
        ax.set_xlabel('Effective Option Spread')
        ax.set_ylabel('Portfolio Buckets')
        ax.set_zlabel('Sharpe Ratio')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        _save_fig(fig_dir, "sharpe_surface_option_underlying_spreads")

        #Get distribution of daily returns for long-short strategy with deciles and not trading costs
        df_truncated = hlPortfolio1.iloc[:len(hlPortfolio1)//13 * 13]
        df_truncated['chunk'] = np.repeat(range(len(hlPortfolio1)//13), 13)
        df_truncated['Returns'] = df_truncated['Returns'] + 1
        df_truncated['Returns'] = df_truncated.groupby('chunk')['Returns'].cumprod()
        hlPortfolio_daily = (df_truncated.groupby('chunk')['Returns'].last() - 1).reset_index()

        #Get distribution of weekly returns for long-short strategy with deciles and not trading costs
        df_truncated = hlPortfolio1.iloc[:len(hlPortfolio1) // 65 * 65]
        df_truncated['chunk'] = np.repeat(range(len(hlPortfolio1) // 65), 65)
        df_truncated['Returns'] = df_truncated['Returns'] + 1
        df_truncated['Returns'] = df_truncated.groupby('chunk')['Returns'].cumprod()
        hlPortfolio_weekly = (df_truncated.groupby('chunk')['Returns'].last() - 1).reset_index()

        #Create boxplot (Figure 4.4)
        CombinedModel.boxplot([hlPortfolio1['Returns'], hlPortfolio_daily['Returns'],
                               hlPortfolio_weekly['Returns']], fig_dir=fig_dir,
                              figure_name="long_short_return_boxplot")

        #Crete performance.xlsx
        writer = pd.ExcelWriter(output_dir / 'performance.xlsx', engine='xlsxwriter')
        evaluationModels.to_excel(writer, sheet_name='Models')
        result_portfolios.to_excel(writer, sheet_name='Ensemble')
        result_portfolios_spread1.to_excel(writer, sheet_name='Ensemble optspreads')
        result_portfolios_spread2.to_excel(writer, sheet_name='Ensemble opt-undspreads')
        hlPortfolio1['Returns'].to_excel(writer, sheet_name='Returns Long-Short 30-min')
        hlPortfolio_daily['Returns'].to_excel(writer, sheet_name='Returns Long-Short Daily')
        hlPortfolio_weekly['Returns'].to_excel(writer, sheet_name='Returns Long-Short Weekly')
        writer.close()

    def shapley_values(self, X_tests, X_trains, fig_dir=None, output_dir=None):
        """
         @brief Main function to calculate the shapley values and create shapley_summary.xlsx
        @param X_tests: Normalized features
        @param X_trains: Features
        """
        def _make_inputs(model_type: str, x_first: torch.Tensor,
                         x_last: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
            if model_type == 'ffn':
                return x_first
            return [x_first, x_last]

        def _make_attention_inputs(x_first: torch.Tensor,
                                   x_last: torch.Tensor) -> list[torch.Tensor]:
            return [x_first.unsqueeze(-1), x_last.unsqueeze(-1)]

        def _format_shap(model_type: str, shap_values) -> np.ndarray:
            if isinstance(shap_values, list):
                if model_type == 'ffn' and len(shap_values) == 1:
                    shap_arr = np.squeeze(np.array(shap_values[0]))
                else:
                    parts = []
                    for el in shap_values:
                        arr = np.squeeze(np.array(el))
                        if arr.ndim == 1:
                            arr = arr.reshape(-1, 1)
                        parts.append(arr)
                    shap_arr = np.concatenate(parts, axis=1) if parts else np.array([])
            else:
                shap_arr = np.squeeze(np.array(shap_values))

            if shap_arr.ndim == 1:
                shap_arr = shap_arr.reshape(-1, 1)
            shap_arr = np.nan_to_num(shap_arr, nan=0.0, posinf=0.0, neginf=0.0)
            return shap_arr

        if output_dir is None:
            output_dir = _infer_output_dir_from_fig_dir(fig_dir) if fig_dir is not None else Path.cwd()
        output_dir = _resolve_output_dir(output_dir)
        fig_dir = _resolve_figure_dir(fig_dir, output_dir=output_dir, subdir="shap/figures")
        np.random.seed(42) #fix seed
        allShaps = []
        x_split_norm = []
        x_split = []
        for i in range(12-self.N, 12): #Choose 1000 contracts per month in the testing set
            indices = np.random.choice(len(X_trains[i]), size=1000, replace=False)
            x_split_norm.append(X_tests[i].iloc[indices,:])
            x_split.append(X_trains[i].iloc[indices,:])

        #Choose 1000 contracts from training set as background
        train = pd.concat(X_trains[:8])
        train_norm = pd.concat(X_tests[:8])
        indices = np.random.choice(len(train), size=1000, replace=False)
        train = train.iloc[indices,:]
        train_norm = train_norm.iloc[indices,:]

        for count, current_model in enumerate(self.model_names): #iterate over model types
            print(f'Shapley values for model: {current_model}')
            if self.normalize[count]: #use normalized data if applicable to model
                x = x_split_norm
                x_train = train_norm
            else:
                x = x_split
                x_train = train

            #create LargeDataset with single batch for background data
            ds_train = LargeDataset(x_train, self.important_columns, self.model_types[count],
                                    oneBatch = True)
            loader = DataLoader(ds_train, batch_size=None)
            x_first_train, x_last_train = next(iter(loader))

            final_shap_values = None
            for i in range(self.N): #Iterate over rolling windows
                shaps = []
                if self.model_types[count] == 'rf': #calculate SHAP values for tree-based models
                    for j in range(self.amount): #iterate over individual models per rolling window
                        # Use TreeExplainer for tree-based models
                        explainer = shap.TreeExplainer(self.totalModels[count][i][j])
                        shap_values = explainer.shap_values(x[i])
                        shap_arr = _format_shap(self.model_types[count], shap_values)
                        shaps.append(shap_arr)

                else: #calculate SHAP values for neural network based methods
                    for j in range(self.amount): #iterate over individual models per rolling window
                        #Create LargeDataset with single batch for testing data
                        ds = LargeDataset(x[i], self.important_columns, self.model_types[count],
                                          oneBatch=True)
                        loader = DataLoader(ds, batch_size=None)
                        x_first, x_last = next(iter(loader))
                        X_tensor = _make_inputs(self.model_types[count], x_first.to('cuda'),
                                                x_last.to('cuda'))
                        # Use model wrapper for individual model to give to Explainer
                        model_wrapper = PyTorchWrapper(self.totalModels[count][i][j],
                                                       self.model_types[count])
                        model_wrapper.eval()

                        if self.model_types[count] == 'attention':
                            Xbg = _make_attention_inputs(x_first_train.to('cuda'),
                                                         x_last_train.to('cuda'))
                            Xte = _make_attention_inputs(x_first.to('cuda'),
                                                         x_last.to('cuda'))
                            shap_arr = None
                            try:
                                explainer = shap.GradientExplainer(model_wrapper, Xbg)
                                shap_values = explainer.shap_values(Xte)
                                if isinstance(shap_values, list):
                                    shap_values = shap_values[0]
                                shap_arr = np.squeeze(np.array(shap_values))
                                if shap_arr.ndim == 1:
                                    shap_arr = shap_arr.reshape(-1, 1)
                                shap_arr = np.nan_to_num(shap_arr, nan=0.0,
                                                         posinf=0.0, neginf=0.0)
                            except Exception:
                                shap_arr = None

                            if (shap_arr is None or not np.isfinite(shap_arr).any() or
                                    np.sum(np.abs(shap_arr)) == 0):
                                x_df = x[i]
                                main_cols = [col for col in x_df.columns
                                             if col not in self.important_columns]
                                context_cols = self.important_columns
                                main_idx = [x_df.columns.get_loc(c) for c in main_cols]
                                context_idx = [x_df.columns.get_loc(c) for c in context_cols]
                                background_size = min(200, len(x_train))
                                bg_idx = np.random.choice(len(x_train), size=background_size,
                                                          replace=False)
                                background_np = x_train.iloc[bg_idx, :].to_numpy()
                                test_np = x_df.to_numpy()

                                bg = np.nan_to_num(background_np.astype(np.float64, copy=True),
                                                   nan=0.0, posinf=0.0, neginf=0.0)
                                te = np.nan_to_num(test_np.astype(np.float64, copy=True),
                                                   nan=0.0, posinf=0.0, neginf=0.0)

                                def predict_fn(X_np: np.ndarray) -> np.ndarray:
                                    X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)
                                    X_np = X_np.astype(np.float32)
                                    x_main = torch.tensor(X_np[:, main_idx],
                                                          dtype=torch.float32, device='cuda')
                                    x_ctx = torch.tensor(X_np[:, context_idx],
                                                         dtype=torch.float32, device='cuda')
                                    x_main = x_main.unsqueeze(-1)
                                    x_ctx = x_ctx.unsqueeze(-1)
                                    model_wrapper.eval()
                                    with torch.no_grad():
                                        y = model_wrapper(x_main, x_ctx).reshape(-1)
                                        y = torch.nan_to_num(y, nan=0.0,
                                                             posinf=0.0, neginf=0.0)
                                        y = torch.clamp(y, -1e6, 1e6)
                                    return y.detach().cpu().numpy().astype(np.float64)

                                try:
                                    explainer = shap.KernelExplainer(predict_fn, bg)
                                    shap_values = explainer.shap_values(te, nsamples=200)
                                    if isinstance(shap_values, list):
                                        shap_values = shap_values[0]
                                    shap_arr = np.squeeze(np.array(shap_values))
                                    if shap_arr.ndim == 1:
                                        shap_arr = shap_arr.reshape(-1, 1)
                                    shap_arr = np.nan_to_num(shap_arr, nan=0.0,
                                                             posinf=0.0, neginf=0.0)
                                except ValueError:
                                    te_subset = te[:min(500, len(te))]
                                    y0 = predict_fn(te_subset)
                                    bg_mean = np.mean(bg, axis=0)
                                    imp_j = np.zeros(te_subset.shape[1], dtype=np.float64)
                                    for col_idx in range(te_subset.shape[1]):
                                        te_j = te_subset.copy()
                                        te_j[:, col_idx] = bg_mean[col_idx]
                                        yj = predict_fn(te_j)
                                        imp_j[col_idx] = np.mean(np.abs(y0 - yj))
                                    imp_j = np.nan_to_num(imp_j, nan=0.0,
                                                          posinf=0.0, neginf=0.0)
                                    shap_arr = np.tile(imp_j, (te.shape[0], 1))
                        else:
                            Xbg = _make_inputs(self.model_types[count],
                                               x_first_train.to('cuda'),
                                               x_last_train.to('cuda'))
                            Xte = X_tensor

                            shap_arr = None
                            try:
                                explainer = shap.DeepExplainer(model_wrapper, Xbg)
                                shap_values = explainer.shap_values(Xte, check_additivity=False)
                                shap_arr = _format_shap(self.model_types[count], shap_values)
                            except Exception:
                                shap_arr = None

                            if (shap_arr is None or not np.isfinite(shap_arr).any() or
                                    np.sum(np.abs(shap_arr)) == 0):
                                try:
                                    explainer = shap.GradientExplainer(model_wrapper, Xbg)
                                    shap_values = explainer.shap_values(Xte)
                                    shap_arr = _format_shap(self.model_types[count], shap_values)
                                except Exception:
                                    shap_arr = None

                            if (shap_arr is None or not np.isfinite(shap_arr).any() or
                                    np.sum(np.abs(shap_arr)) == 0):
                                x_df = x[i]
                                main_cols = [col for col in x_df.columns
                                             if col not in self.important_columns]
                                context_cols = self.important_columns
                                main_idx = [x_df.columns.get_loc(c) for c in main_cols]
                                context_idx = [x_df.columns.get_loc(c) for c in context_cols]
                                background_size = min(200, len(x_train))
                                bg_idx = np.random.choice(len(x_train), size=background_size,
                                                          replace=False)
                                background_np = x_train.iloc[bg_idx, :].to_numpy()
                                test_np = x_df.to_numpy()

                                def predict_fn(X_np: np.ndarray) -> np.ndarray:
                                    with torch.no_grad():
                                        if self.model_types[count] == 'ffn':
                                            x_main = torch.tensor(X_np, dtype=torch.float32,
                                                                  device='cuda')
                                            y = model_wrapper(x_main)
                                        else:
                                            x_main = torch.tensor(X_np[:, main_idx],
                                                                  dtype=torch.float32,
                                                                  device='cuda')
                                            x_ctx = torch.tensor(X_np[:, context_idx],
                                                                 dtype=torch.float32,
                                                                 device='cuda')
                                            if self.model_types[count] == 'attention':
                                                x_main = x_main.unsqueeze(-1)
                                                x_ctx = x_ctx.unsqueeze(-1)
                                            y = model_wrapper(x_main, x_ctx)
                                        return y.detach().cpu().numpy().reshape(-1)

                                explainer = shap.KernelExplainer(predict_fn, background_np)
                                shap_values = explainer.shap_values(test_np, nsamples=200)
                                if isinstance(shap_values, list):
                                    shap_values = shap_values[0]
                                shap_arr = _format_shap(self.model_types[count], shap_values)

                        shaps.append(shap_arr)

                # Average SHAP values over individual models per rolling window
                shap_values = sum(shaps) / len(shaps)
                # Concat SHAP values for rolling windows (safe concat as None possible)
                final_shap_values = CombinedModel.safe_concatenate(final_shap_values, shap_values)

            if self.normalize[count]: #unnormalize returns if model was trained with normalized data
                normalizations = pd.read_excel('./normalization_params.xlsx', index_col=0)
                final_shap_values = (final_shap_values * normalizations.loc['std', 'Returns'])
            allShaps.append(final_shap_values)

        #Average SHAP values over model types using the respective weights (if we evaluate Ensemble model)
        shap_values = self.model_weights[0] * allShaps[0]
        for i in range(1, len(self.model_names)):
            shap_values += (self.model_weights[i] * allShaps[i])

        x_concat = pd.concat(x_split) #concat data over rolling windows
        feature_names = x_concat.columns.tolist() #get names of features in dataset

        #Calculate SHAP values over different samples and their contribution to all features
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        # bars in Figure 4.5 (percentage SHAP values averaged)
        denom = np.sum(mean_abs_shap)
        if (not np.isfinite(denom)) or denom <= 0:
            denom = 1.0
        percentage_shap = mean_abs_shap / denom
        # points in Figure 4.5 (percentage SHAP values of individual contracts)
        row_denom = np.sum(np.abs(shap_values), axis=1, keepdims=True)
        row_denom = np.where((~np.isfinite(row_denom)) | (row_denom <= 0), 1.0, row_denom)
        percentage_shap_individual = np.abs(shap_values) / row_denom

        #Get top 10 features
        top_indices = np.argsort(percentage_shap)[-10:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_shap_vals = shap_values[:, top_indices]
        top_feat_vals = x_concat.iloc[:, top_indices]

        #Plot Figure 4.5
        for i, idx in enumerate(top_indices):
            #Bar
            plt.barh(i, percentage_shap[idx], color=plt.get_cmap('tab10')(i))
            #Points
            x = percentage_shap_individual[:, idx]
            y = np.random.normal(i, 0.05, size=len(x))  # jitter
            plt.scatter(x, y, alpha=0.7, s=15, color=plt.get_cmap('tab10')(i), edgecolors='k', linewidths=0.1)
        plt.yticks(range(10), top_features)
        plt.xlabel("SHAP Value")
        plt.axvline(0, color='gray', linewidth=0.5)
        plt.tight_layout()
        _save_fig(fig_dir, f"shap_importance_{current_model}", dpi=200)

        #Plot Figure 4.6
        cmap = plt.get_cmap('viridis')
        fig, ax = plt.subplots()
        for i in range(len(top_indices)):
            y = top_shap_vals[:, i]
            feat_vals = top_feat_vals.iloc[:, i]
            #Normalize feature values for colormap mapping
            norm = mcolors.Normalize(vmin=np.nanpercentile(feat_vals, 5),
                                     vmax=np.nanpercentile(feat_vals, 95))
            colors = cmap(norm(feat_vals))
            #Jitter x-axis
            x = np.random.normal(i, 0.05, size=len(y))
            plt.scatter(y, x, c=colors, s=15, edgecolor='k', linewidth=0.1)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_xlabel("Change in E[r]")

        #Add colorbar for reference only
        cbar_ax = fig.add_axes([0.87, 0.125, 0.015, 0.82])  #[left, bottom, width, height]
        norm_dummy = mcolors.Normalize(vmin=0, vmax=1)
        cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm_dummy, orientation='vertical')
        cb.set_label("Feature Value")
        cbar_ax.set_yticks([0, 1])
        cbar_ax.set_yticklabels(['Low', 'High'])

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        _save_fig(fig_dir, f"shap_beeswarm_{current_model}", dpi=200)

        #Create shapley_summary.xlsx
        df = pd.DataFrame({
            "Feature": feature_names,
            "Mean |SHAP|": list(mean_abs_shap),
            "Percentage of Importance": list(percentage_shap)
        })
        #Sort by percentage importance
        df = df.sort_values("Percentage of Importance", ascending=False)
        df.to_excel(output_dir / "shapley_summary.xlsx", index=False)

    def run_diebold_mariano_matrix(self, x_cleans_normalized: list[pd.DataFrame],
                                   x_cleans_normalized_ffn: list[pd.DataFrame],
                                   x_cleans: list[pd.DataFrame], xs: list[pd.DataFrame],
                                   ensembles: list[str]) -> None:
        """
        @brief Main function to initiate creation of diebold mariano matrix (Table 4.2)
            Conducts diebold-mariano forecast comparison between all model combinations
        @param x_cleans_normalized: Normalized features
        @param x_cleans_normalized_ffn: Normalized features for ffn
        @param x_cleans: Features
        @param xs: Data including further information such as realized returns, etc.
        @param ensembles: List of models to also combine into ensemble
        """
        model_labels = list(self.model_names) + ['Ensemble']
        print(f"[INFO] DM model labels: {model_labels}")

        # Create aligned square dataframes for statistics and p-values.
        results_stat = pd.DataFrame(index=model_labels, columns=model_labels, dtype=float)
        results_p = pd.DataFrame(index=model_labels, columns=model_labels, dtype=float)
        print(f"[INFO] DM statistic matrix shape: {results_stat.shape}")
        print(f"[INFO] DM p-value matrix shape: {results_p.shape}")
        assert list(results_stat.index) == list(results_stat.columns), (
            "Statistic matrix row and column labels do not match."
        )
        assert list(results_p.index) == list(results_p.columns), (
            "p-value matrix row and column labels do not match."
        )

        # Perform prediction for individual models and ensemble.
        y_preds = self.predict(x_cleans_normalized, x_cleans, x_cleans_normalized_ffn, ensembles)
        pred_lengths = [len(np.asarray(pred).reshape(-1)) for pred in y_preds]
        print(f"[INFO] DM prediction count: {len(y_preds)}")
        print(f"[INFO] DM prediction lengths: {pred_lengths}")
        assert len(y_preds) == len(model_labels), (
            f"Number of predictions ({len(y_preds)}) does not match number of labels "
            f"({len(model_labels)})."
        )

        x_long = pd.concat(xs).copy()
        for i, row_model in enumerate(model_labels): #iterate over model types
            for j, col_model in enumerate(model_labels): #iterate over model types (create matrix)

                x_long['Prediction1'] = y_preds[i] #prediction row model
                x_long['Prediction2'] = y_preds[j] #prediction column model

                grouped = x_long.groupby(['loctimestamp', 'time']) #group data over timepoints
                cs = []
                for (loctimestamp, time), group in grouped: #iterate over timepoints
                    n = len(group)
                    if n == 0:
                        result = 0
                    else:
                        if i == j:
                            #c for Clark-West test statistic (each timepoint represents one c)
                            # (If same models perform Clark-West test)
                            result = (1 / n) * (((group['Returns']) ** 2 - (group['Returns'] -
                                                group['Prediction2']) ** 2).sum())
                        else:
                            #d for Diebold-Mariano test statistic (each timepoint represents one d)
                            result = (1 / n) * (((group['Returns'] - group['Prediction1']) ** 2 -
                                    (group['Returns'] - group['Prediction2']) ** 2).sum())
                    cs.append(result)

                if i != j: #Perform calculation of Diebold-Mariano test statistic using 21*13 lags
                    dm_stat, dm_p = CombinedModel.diebold_mariano_test(cs, 21 * 13)
                else: #Perform calculation of Clark-West test statistic using 21*13 lags
                    dm_stat, dm_p = CombinedModel.diebold_mariano_test(cs, 21 * 13, 'greater')

                #add statistic and p-value to dfs
                results_stat.loc[row_model, col_model] = dm_stat
                results_p.loc[row_model, col_model] = dm_p

        dm_values = results_stat.to_numpy(dtype=float)
        upper_idx = np.triu_indices_from(dm_values, k=1)
        antisym_errors = np.abs(dm_values + dm_values.T)[upper_idx]
        max_antisym_error = float(np.nanmax(antisym_errors)) if antisym_errors.size else 0.0
        print(f"[INFO] DM max antisymmetry error: {max_antisym_error}")

        #Create excel file
        with pd.ExcelWriter('../analysis/diebold_mariano.xlsx', engine='xlsxwriter') as writer:
            results_stat.to_excel(
                writer, sheet_name='Statistic', index=True, index_label='Model'
            )
            results_p.to_excel(
                writer, sheet_name='p-Value', index=True, index_label='Model'
            )

            workbook = writer.book
            header_format = workbook.add_format({
                'bold': True,
                'align': 'center',
                'valign': 'vcenter',
                'border': 1
            })
            index_format = workbook.add_format({
                'bold': True,
                'align': 'left',
                'valign': 'vcenter',
                'border': 1
            })
            number_format = workbook.add_format({
                'num_format': '0.000000',
                'align': 'center',
                'valign': 'vcenter'
            })

            for sheet_name, frame in [('Statistic', results_stat), ('p-Value', results_p)]:
                worksheet = writer.sheets[sheet_name]
                worksheet.freeze_panes(1, 1)
                worksheet.set_row(0, None, header_format)

                index_width = max(
                    len('Model'),
                    max(len(str(label)) for label in frame.index)
                ) + 2
                worksheet.set_column(0, 0, index_width, index_format)

                for col_idx, col_label in enumerate(frame.columns, start=1):
                    series = frame[col_label]
                    column_width = max(
                        len(str(col_label)),
                        max(len(f"{value:.6f}") for value in series if pd.notna(value))
                        if series.notna().any() else 0
                    ) + 2
                    worksheet.set_column(col_idx, col_idx, column_width, number_format)

        print("[OK] Clean Diebold-Mariano matrix written to ../analysis/diebold_mariano.xlsx")

    @staticmethod
    def safe_concatenate(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
        """
        @brief Method to safely concatenate two arrays even if one is None
        @param a: First array
        @param b: Second array
        @return: Concatenated array or None if both are None
        """
        if a is None and b is None:
            return None
        elif a is None:
            return b
        elif b is None:
            return a
        else:
            return np.concatenate((a, b), axis=0)

    @staticmethod
    def boxplot(dfs: list[pd.DataFrame], fig_dir=None,
                figure_name: str = "return_boxplot_intervals") -> None:
        """
        @brief Create boxplot for distribution of returns for longer periods (Figure 4.4)
        @param dfs: List of dataframes to create boxplot
        """
        if len(dfs) != 3 or any(len(pd.Series(scores).dropna()) == 0 for scores in dfs):
            print(f'Warning: skipped {figure_name} because one or more return series were empty.')
            return

        fig_dir = _resolve_figure_dir(fig_dir)
        plt.rcParams.update({'font.size': 22})
        #Function to compute minimum, 25th, 50th, 75th percentiles and maximum
        def custom_boxplot_stats(series):
            return {
                'whislo': min(series),
                'q1': np.percentile(series, 25),
                'med': np.median(series),
                'q3': np.percentile(series, 75),
                'whishi': max(series),
            }

        plt.figure()
        colors = sns.color_palette("Spectral", 13) if sns is not None else ["#9e0142", "#fdae61", "#5e4fa2"]
        for i in range(3): #iterate over three periods (30-min, daily, weekly)
            scores = pd.Series(dfs[i]).dropna()
            stats = custom_boxplot_stats(scores)

            #Plot box
            plt.plot([i, i], [stats['whislo'], stats['q1']], color='k')  #lower whisker
            plt.plot([i - 0.2, i + 0.2], [stats['whislo'], stats['whislo']], color='k',
                     linewidth=1.2) #lower whisker
            plt.plot([i, i], [stats['whishi'], stats['q3']], color='k')  #upper whisker
            plt.plot([i - 0.2, i + 0.2], [stats['whishi'], stats['whishi']], color='k',
                     linewidth=1.2) #upper whisker
            plt.fill_betweenx(
                [stats['q1'], stats['q3']], i - 0.4, i + 0.4,
                color=colors[i], edgecolor='k', linewidth=1.2
            ) #box
            # median
            plt.plot([i - 0.4, i + 0.4], [stats['med'], stats['med']], color='k', linewidth=1.2)
            # mean
            plt.plot(i, np.mean(scores), 'o', color='white', markersize=6, markeredgecolor='k')

        #Customize plot
        plt.xticks(range(3), ['30-min', 'Daily', 'Weekly'])
        plt.axhline(0, linestyle='--', color='black', linewidth=0.8)
        plt.ylabel(r"Delta-hedged Return")
        plt.xlabel('Interval')
        plt.tight_layout()
        _save_fig(fig_dir, figure_name)

class PyTorchWrapper(torch.nn.Module):
    """
    @brief Wrapper Class for PyTorch models to give to shap Explainer
    """
    def __init__(self, model, model_type: str) -> None:
        """
        @brief Initialize PyTorch wrapper class
        @param model: Model as defined in model.py
        @param model_type: Type of model
        """
        super().__init__()
        self.model = model
        self.model_type = model_type

    def forward(self, *x) -> torch.Tensor:
        """
        @brief Forward pass of PyTorch model
        :param x: Input
        :return: Output
        """
        if self.model_type == 'autoencoder': #Use forward pass for inference for autoencoder
            return self.model.validate(x[0], x[1])
        if self.model_type in ['fusion', 'fusionContextFirst']: #multiply prediction for simple fusion models
            if self.model_type == 'fusionContextFirst':
                x_main, x_ctx = x[1], x[0]
            else:
                x_main, x_ctx = x[0], x[1]
            main_outputs = []
            for sub_model in self.model[0].models:
                main_outputs.append(sub_model(x_main))
            main_pred = sum(main_outputs) / len(main_outputs)
            return main_pred * self.model[1](x_ctx)
        if self.model_type == 'ffn': #for ffn all data are main features
            return self.model(x[0])
        else: #pass main and modulating features
            return self.model(x[0], x[1])
