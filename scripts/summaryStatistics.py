"""
@file summaryStatistics.py
@brief Create summary statistics for the dataset

This script creates the summaryStatistics.xlsx file that contains statistical information about
each feature for the whole dataset or parts of the dataset (Table 3.2). Additionally statistics for
returns at different times of the day are created and visualized in a boxplot (Table 3.3 and Figure 3.10).

@details
Functions:
- preprocess(all_Call_Put) - Loads the full dataset
- boxplot(df) - Creates a boxplot from given dataframe
- create_summary(data) - Creates summary statistics for given df
- run_summary_creation() - Runs creation of multiple different summaries
"""

import os
os.environ["MPLBACKEND"] = "Agg"

from pathlib import Path

from plot_saver import save_current_figure

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera

#Some params for plotting
plt.rcParams["figure.figsize"] = (16,9)
plt.rcParams.update({'font.size': 22})

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def preprocess(all_Call_Put: int) -> list[pd.DataFrame]:
    """
    @brief Loads the full dataset
    @param all_Call_Put: keeps all features (0), only calls (1) or only puts (2)
    @return: list of dfs for each month
    """
    dfs = []
    for i in range(1, 13):
        df = pd.read_csv(f'../data/data_month_{i}.csv')
        if all_Call_Put == 1:
            df = df[df['putcall_P']==0]
        elif all_Call_Put == 2:
            df = df[df['putcall_P']==1]
        float_cols = df.select_dtypes(include=['float']).columns
        df[float_cols] = df[float_cols].astype(np.float32)
        int_cols = df.select_dtypes(include=['int']).columns
        df[int_cols] = df[int_cols].astype(np.int32)
        df.drop(columns=['Unnamed: 0'], inplace=True)
        dfs.append(df)
    return dfs

def boxplot(df: pd.DataFrame) -> None:
    """
    @brief Creates boxplot of returns for each time of day for given df (Figure 3.10)
    @param df: df to create boxplot for
    """
    #Function to compute 5th, 25th, 50th, 75th and 95th percentiles
    def custom_boxplot_stats(series):
        return {
            'whislo': np.percentile(series, 5),
            'q1': np.percentile(series, 25),
            'med': np.median(series),
            'q3': np.percentile(series, 75),
            'whishi': np.percentile(series, 95),
        }

    plt.figure()
    colors = sns.color_palette("Spectral", 13)
    for i in range(13): #Go through each time of day
        scores = df.iloc[:, i] #Take i-th time only
        scores = scores.dropna()
        stats = custom_boxplot_stats(scores)

        #Draw plot
        plt.plot([i, i], [stats['whislo'], stats['q1']], color='k') #lower whisker
        plt.plot([i - 0.2, i + 0.2], [stats['whislo'], stats['whislo']], color='k',
                 linewidth=1.2) #lower whisker
        plt.plot([i, i], [stats['whishi'], stats['q3']], color='k') #higher whisker
        plt.plot([i - 0.2, i + 0.2], [stats['whishi'], stats['whishi']], color='k',
                 linewidth=1.2) #higher whisker
        plt.fill_betweenx(
            [stats['q1'], stats['q3']], i - 0.4, i + 0.4,
            color=colors[i], edgecolor='k', linewidth=1.2
        ) #colored box
        # median
        plt.plot([i - 0.4, i + 0.4], [stats['med'], stats['med']], color='k', linewidth=1.2)
        # mean
        plt.plot(i, np.mean(scores), 'o', color='white', markersize=6, markeredgecolor='k')

    #Customize plot
    plt.xticks(range(13), [str(j) for j in range(1,14)])
    plt.axhline(0, linestyle='--', color='black', linewidth=0.8)
    plt.ylabel(r"Delta-hedged Return")
    plt.xlabel('Time of Day')
    plt.tight_layout()
    save_current_figure(FIG_DIR, "time_of_day_boxplot")

def create_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Creates summary statistics for given data (used in Table 3.2 and Table 3.3)
    @param data: df to create summary statistics for
    @return: summary statistics for given df
    """
    summary = data.describe(include="all").transpose()
    summary['skew'] = data.skew(numeric_only=True)
    summary['kurtosis'] = data.kurtosis(numeric_only=True)
    summary['JB'] = None
    # Perform Jarque-Bera test only for returns
    summary.loc['Returns', 'JB'] = jarque_bera(data['Returns'])[1]
    summary["missing_values"] = data.isnull().sum()
    summary["unique_values"] = data.nunique()
    return summary

def run_summary_creation() -> None:
    """
    @brief Create multiple summary statistics for dataset
    """

    #Create summary for full dataset
    dfs = preprocess(0)
    data = pd.concat(dfs)
    print(f'All: {data.info()}')
    summary = create_summary(data)
    timeReturns = data[['Returns', 'time']]
    del data

    #Create return summary for each time of day
    rets = {}
    rets_zoomed = {}
    points = ['10:00-10:30', '10:30-11:00', '11:00-11:30', '11:30-12:00', '12:00-12:30', '12:30-13:00',
              '13:00-13:30', '13:30-14:00', '14:00-14:30', '14:30-15:00', '15:00-15:30', '15:30-16:00',
              '16:00-16:15']
    for i in range(13):
        ret = timeReturns[timeReturns['time'] == i]['Returns']
        rets[f'{points[i]}'] = ret.reset_index(drop=True)

    new_df = pd.DataFrame(rets)
    new_df = new_df.reset_index(drop=True)
    summary_time = new_df.describe(include="all").transpose()
    summary_time['skew'] = new_df.skew(numeric_only=True)
    summary_time['kurtosis'] = new_df.kurtosis(numeric_only=True)
    summary_time["missing_values"] = new_df.isnull().sum()
    summary_time["unique_values"] = new_df.nunique()

    #Create summary for train dataset
    data = pd.concat(dfs[:9])
    print(f'All Train: {data.info()}')
    summary_train = create_summary(data)
    del data

    #Create summary for test dataset
    data = pd.concat(dfs[9:])
    print(f'All Test: {data.info()}')
    summary_test = create_summary(data)
    del data
    del dfs

    #Create summary for only calls
    dfs = preprocess(1)
    data = pd.concat(dfs)
    print(f'Call: {data.info()}')
    summary_call = create_summary(data)
    del data
    del dfs

    #Create summary for only puts
    dfs = preprocess(2)
    data = pd.concat(dfs)
    print(f'Put: {data.info()}')
    summary_put = create_summary(data)
    del data
    del dfs

    # Create boxplot for time of day returns
    boxplot(new_df)

    # Create excel file with statistics
    with pd.ExcelWriter('../analysis/summary_statistics.xlsx') as writer:
        summary.to_excel(writer, sheet_name='All')
        summary_train.to_excel(writer, sheet_name='All_Train')
        summary_test.to_excel(writer, sheet_name='All_Test')
        summary_call.to_excel(writer, sheet_name='Call')
        summary_put.to_excel(writer, sheet_name='Put')
        summary_time.to_excel(writer, sheet_name='Timepoints')

if __name__ == "__main__":
    run_summary_creation()
