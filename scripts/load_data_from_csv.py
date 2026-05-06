"""
@file load_data_from_csv.py
@brief Loads the data from multiple csv files and processes it

This script performs the preprocessing of the available data from multiple csv files, that represent
tables in the database. This includes calculating the delta-hedged returns,
adding features not yet implemented, removing features that are faulty. The data is saved in monthly files
and a short summary is created for each month.

@details
Values to set manually:
- INTERPOLATE - As in some of the later datatables features are only available for specific maturities,
    for contracts with different maturities we can either use the closest available features or
    interpolate them. Interpolation however is slow, but used in the thesis as more accurate
- MATURITIES - Maturities for which features are available in all tables (fixed for thesis)

Functions:
- load_and_process_data() - Loads all csv files, processes the data, calculates delta-hedged returns
    and creates the monthly output files
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

OUT_DIR = Path("./data/month")
OUT_DIR.mkdir(parents=True, exist_ok=True)

#As in some of the later datatables features are only available for specific maturities,
# for contracts with different maturities we can either use the closest available features or
# interpolate them. Slow if True
INTERPOLATE = True
MATURITIES = [1,2,3,5,10,30,60,91,365,730,1095] #Maturities for which features are available in all tables

def load_and_process_data() -> None:
    """
    @brief Loads all the data from multiple csv files and processes it
    """
    #Load data from first file
    df = pd.read_csv('./data/OptionData_PricesToVolas_VIEW_SPX_2015.csv', low_memory=False)
    df = df.drop(columns=['underlyingtype', 'optionstyle', 'earlyExercisePremium']) #Remove constant features
    #Split timestamp into date and time
    df[['loctimestamp', 'time']] = df['loctimestamp'].str.split(' ', n=1, expand=True)

    #Load data from second file
    df2 = pd.read_csv('./data/CBOE_30min_VIEW_SPX_2015.csv', low_memory=False)
    #Split timestamp into date and time
    df2[['loctimestamp', 'time']] = df2['quote_datetime'].str.split(' ', n=1, expand=True)
    #Calculate the difference in days and create a new column 'daystomaturity' (missing in original file)
    df2['start_date'] = pd.to_datetime(df2['loctimestamp'])
    df2['end_date'] = pd.to_datetime(df2['expiration'])
    df2['daystomaturity'] = (df2['end_date'] - df2['start_date']).dt.days
    #Rename features and remove constant features or features that are redundant and taken from another file
    df2 = df2.rename(columns={'option_type': 'putcall'})
    df2 = df2.drop(columns=['underlying_symbol', 'root', 'open', 'high', 'low', 'close',
                            'trade_volume', 'active_underlying_price', 'open_interest', 'quote_datetime',
                            'start_date', 'end_date', 'expiration'])

    #Load data from third file
    df3 = pd.read_csv('./data/RNM_VIEW_SPX_2015.csv', low_memory=False)
    #Remove constant features or features that are redundant and taken from another file
    df3 = df3.drop(columns=['instrumentid', 'underlyingprice', 'bakshiImplVar_eps',
                            'bakshiImplVol_eps_annual', 'bakshiSkew_eps', 'bakshiKurt_eps', 'implBeta'])
    #Split timestamp into date and time
    df3[['loctimestamp', 'time']] = df3['loctimestamp'].str.split(' ', n=1, expand=True)

    #Load data from fourth file
    df4 = pd.read_csv('./data/P_implQMeasures_VIEW_SPX_2015.csv', low_memory=False)
    #Remove constant features or features that are redundant and taken from another file
    df4 = df4.drop(columns=['instrumentid', 'underlyingprice', 'underlyingforwardprice',
                            'P_implQ_implVol_annual', 'P_implQ_beta', 'P_implQ_mu_eps',
                            'P_implQ_var_eps', 'P_implQ_implVol_sys_annual', 'P_implQ_implVol_eps_annual',
                            'P_implQ_skew_eps', 'P_implQ_kurt_eps',
                            'P_implQ_std_norm_eps_5', 'P_implQ_std_norm_eps_6', 'P_implQ_std_norm_eps_7',
                            'P_implQ_std_norm_eps_8', 'P_implQ_std_norm_eps_9', 'P_implQ_std_norm_eps_10'])
    #Remove features that are almost all None after merging with other dfs
    df4 = df4.drop(columns=['VaRP_implQ_pct_0_1', 'VaRP_implQ_pct_1', 'VaRP_implQ_pct_5',
                            'VaRP_implQ_pct_25', 'VaRP_implQ_pct_50', 'VaRP_implQ_pct_75',
                            'VaRP_implQ_pct_95', 'VaRP_implQ_pct_99', 'VaRP_implQ_pct_99_9',
                            'AVaRP_implQ_pct_0_1', 'AVaRP_implQ_pct_1', 'AVaRP_implQ_pct_5',
                            'AVaRP_implQ_pct_95', 'AVaRP_implQ_pct_99', 'AVaRP_implQ_pct_99_9'])
    #Split timestamp into date and time
    df4[['loctimestamp', 'time']] = df4['loctimestamp'].str.split(' ', n=1, expand=True)
    #Split features that contain multiple features and are given as list
    y = df4['P_implQ_moments_1_10'].apply(lambda x: x.split(';'))
    new_cols = pd.DataFrame(y.tolist(), columns=[f'P_implQ_moments_{i}' for i in range(1, 11)], dtype=float)
    df4 = pd.concat([df4, new_cols], axis=1)
    y = df4['P_implQ_cumulants_1_10'].apply(lambda x: x.split(';'))
    new_cols = pd.DataFrame(y.tolist(), columns=[f'P_implQ_cumulants_{i}' for i in range(1, 11)], dtype=float)
    df4 = pd.concat([df4, new_cols], axis=1)
    y = df4['P_implQ_cumulants_sys_1_10'].apply(lambda x: x.split(';'))
    new_cols = pd.DataFrame(y.tolist(), columns=[f'P_implQ_cumulants_sys_{i}' for i in range(1, 11)], dtype=float)
    df4 = pd.concat([df4, new_cols], axis=1)
    df4 = df4.drop(columns=['SP_implQ_alpha_0_1_to_99_9_delta_0_1', 'P_implQ_moments_1_10',
                            'P_implQ_cumulants_1_10', 'P_implQ_cumulants_sys_1_10',
                            'P_implQ_cumulants_eps_1_10'])

    #Load data from fith file
    timeTranslation = {"09:59:59": "10:00:00", "10:29:59": "10:30:00", "10:59:59": "11:00:00",
                       "11:29:59": "11:30:00", "11:59:59": "12:00:00", "12:29:59": "12:30:00",
                       "12:59:59": "13:00:00", "13:29:59": "13:30:00", "13:59:59": "14:00:00",
                       "14:29:59": "14:30:00", "14:59:59": "15:00:00", "15:29:59": "15:30:00",
                       "15:59:59": "16:00:00", "16:14:59": "16:15:00"}
    df5 = pd.read_csv('./data/P_MeasuresIntraday_VIEW_SPX_2015.csv', low_memory=False)
    #Remove constant features or features that are redundant and taken from another file
    df5 = df5.drop(columns=['instrumentid', 'loctimestamp_start', 'freq', 'subsampling_delta',
                            'num_price_obs', 'r_corr', 'P_realizedVol_annual', 'P_alpha', 'P_beta',
                            'P_mu_eps', 'P_var_eps', 'P_realizedVol_sys_annual', 'P_realizedVol_eps_annual',
                            'P_skew_eps', 'P_kurt_eps',
                            'P_std_norm_eps_5', 'P_std_norm_eps_6', 'P_std_norm_eps_7', 'P_std_norm_eps_8',
                            'P_std_norm_eps_9', 'P_std_norm_eps_10'])
    #Remove features that are almost all None after merging with other dfs
    df5 = df5.drop(columns=['VaRP_pct_0_1', 'VaRP_pct_1', 'VaRP_pct_5', 'VaRP_pct_25', 'VaRP_pct_50',
                            'VaRP_pct_75', 'VaRP_pct_95', 'VaRP_pct_99', 'VaRP_pct_99_9', 'AVaRP_pct_0_1',
                            'AVaRP_pct_1', 'AVaRP_pct_5', 'AVaRP_pct_95', 'AVaRP_pct_99', 'AVaRP_pct_99_9'])
    #Split timestamp into date and time and round up endtimes
    locts_parts = df5['loctimestamp_end'].str.split(' ', n=1, expand=True)
    df5['loctimestamp_end'] = locts_parts[0]
    df5['time'] = locts_parts[1].str.split('.', n=1, expand=True)[0]
    mapped_time = df5['time'].map(timeTranslation)
    if mapped_time.isna().any():
        missing = df5.loc[mapped_time.isna(), 'time'].unique().tolist()
        raise KeyError(f"Missing time translation for: {missing}")
    df5['time'] = mapped_time
    df5 = df5.rename(columns={'loctimestamp_end': 'loctimestamp'})
    #Split features that contain multiple features and are given as list
    y = df5['P_moments_1_10'].apply(lambda x: x.split(';'))
    new_cols = pd.DataFrame(y.tolist(), columns=[f'P_moments_{i}' for i in range(1, 11)], dtype=float)
    df5 = pd.concat([df5, new_cols], axis=1)
    y = df5['P_cumulants_1_10'].apply(lambda x: x.split(';'))
    new_cols = pd.DataFrame(y.tolist(), columns=[f'P_cumulants_{i}' for i in range(1, 11)], dtype=float)
    df5 = pd.concat([df5, new_cols], axis=1)
    y = df5['P_cumulants_sys_1_10'].apply(lambda x: x.split(';'))
    new_cols = pd.DataFrame(y.tolist(), columns=[f'P_cumulants_sys_{i}' for i in range(1, 11)], dtype=float)
    df5 = pd.concat([df5, new_cols], axis=1)
    df5 = df5.drop(columns=['P_moments_1_10', 'P_cumulants_1_10', 'P_cumulants_sys_1_10',
                            'P_cumulants_eps_1_10', 'SP_alpha_0_1_to_99_9_delta_0_1'])

    #Load data from sixth file
    df6 = pd.read_csv('./data/RNDImplMeasures_VIEW_SPX_2015.csv', low_memory=False)
    #Remove constant features or features that are redundant and taken from another file
    df6 = df6.drop(columns=['instrumentid', 'underlyingprice', 'underlyingforwardprice',
                            'RND_beta', 'RND_mu_eps', 'RND_var_eps', 'RND_implVol_eps_annual',
                            'RND_skew_eps', 'RND_kurt_eps', 'RND_std_norm_eps_5', 'RND_std_norm_eps_6',
                            'RND_std_norm_eps_7', 'RND_std_norm_eps_8',
                            'RND_std_norm_eps_9', 'RND_std_norm_eps_10'])
    #Split timestamp into date and time
    df6[['loctimestamp', 'time']] = df6['loctimestamp'].str.split(' ', n=1, expand=True)
    #Split features that contain multiple features and are given as list
    y = df6['RND_moments_0_10'].apply(lambda x: x.split(';'))
    new_cols = pd.DataFrame(y.tolist(), columns=[f'RND_moments_{i}' for i in range(0, 11)], dtype=float)
    df6 = pd.concat([df6, new_cols], axis=1)
    y = df6['RND_cumulants_1_10'].apply(lambda x: x.split(';'))
    new_cols = pd.DataFrame(y.tolist(), columns=[f'RND_cumulants_{i}' for i in range(1, 11)], dtype=float)
    df6 = pd.concat([df6, new_cols], axis=1)
    y = df6['RND_cumulants_sys_1_10'].apply(lambda x: x.split(';'))
    new_cols = pd.DataFrame(y.tolist(), columns=[f'RND_cumulants_sys_{i}' for i in range(1, 11)], dtype=float)
    df6 = pd.concat([df6, new_cols], axis=1)
    y = df6['SQ_alpha_0_1_to_99_9_delta_0_1'].apply(lambda x: x.split(';'))
    # This feature originally contains 1000 features. To minimize the data we only use specific ones
    selected_cols = pd.DataFrame(y.apply(lambda x: [x[0], x[9], x[49], x[249], x[499], x[749], x[949],
                                                    x[989], x[-1]]).tolist(),
                                 columns=['SQ_alpha_0_1', 'SQ_alpha_1', 'SQ_alpha_5', 'SQ_alpha_25',
                                          'SQ_alpha_50', 'SQ_alpha_75', 'SQ_alpha_95', 'SQ_alpha_99',
                                          'SQ_alpha_99_9'], dtype=float)
    df6 = pd.concat([df6, selected_cols], axis=1)
    df6 = df6.drop(columns=['SQ_alpha_0_1_to_99_9_delta_0_1', 'RND_moments_0_10', 'RND_cumulants_1_10',
                            'RND_cumulants_sys_1_10', 'RND_cumulants_eps_1_10'])

    #Further processing
    # Convert all float columns to float32
    float_cols = df.select_dtypes(include=['float']).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    float_cols = df2.select_dtypes(include=['float']).columns
    df2[float_cols] = df2[float_cols].astype(np.float32)

    float_cols = df3.select_dtypes(include=['float']).columns
    df3[float_cols] = df3[float_cols].astype(np.float32)

    float_cols = df4.select_dtypes(include=['float']).columns
    df4[float_cols] = df4[float_cols].astype(np.float32)

    float_cols = df5.select_dtypes(include=['float']).columns
    df5[float_cols] = df5[float_cols].astype(np.float32)

    float_cols = df6.select_dtypes(include=['float']).columns
    df6[float_cols] = df6[float_cols].astype(np.float32)

    #Merge df1 and df2 and remove options with too high spread and other anomalies (ask < bid, etc.)
    df = df.merge(df2, on=['time', 'loctimestamp', 'daystomaturity', 'strike', 'putcall'], how='inner')
    del df2
    df['spread_size'] = (df['ask'] - df['bid'])/df['price']
    df = df[df['spread_size'] <= 0.1]
    df = df.drop(columns=['spread_size'])
    df = df[df['bid'] > 0]
    df = df[df['ask'] > df['bid']]
    df = df[df['price'] >= 0.125]

    #Drop all contracts without a delta
    df = df.dropna(subset=['delta'])
    df.reset_index(drop=True, inplace=True)

    #Calculation of delta-hedged returns
    #Apply time mapping to get values in next period
    unique_times = ['10:00:00', '10:30:00', '11:00:00', '11:30:00', '12:00:00', '12:30:00',
                    '13:00:00', '13:30:00', '14:00:00', '14:30:00', '15:00:00', '15:30:00',
                    '16:00:00', '16:15:00'] #Trading times
    time_mapping = {time: i for i, time in enumerate(unique_times)}
    df['time'] = df['time'].map(time_mapping)
    df3['time'] = df3['time'].map(time_mapping)
    df4['time'] = df4['time'].map(time_mapping)
    df5['time'] = df5['time'].map(time_mapping)
    df6['time'] = df6['time'].map(time_mapping)
    df = df.sort_values(by='time')

    #Merge df with itself to find next time step
    df_next = df.copy()
    df_next['time'] -= 1  #Shift time values to match with previous ones
    df_merged = df.merge(df_next,
                         on=['loctimestamp', 'daystomaturity', 'strike', 'putcall', 'time'],
                         suffixes=('', '_next'))
    del df_next
    #Calculate returns according to formula
    df_merged['Returns'] = (df_merged['price_next'] - df_merged['price'] - df_merged['delta']*
                            (df_merged['underlyingprice_next'] - df_merged['underlyingprice']) -
                            (df_merged['riskfree']/(252*13))*(df_merged['price'] - df_merged['delta']*
                            df_merged['underlyingprice'])) / (np.abs(df_merged['price'] -
                            df_merged['delta']*df_merged['underlyingprice']))

    #Keep certain infos for next period for later evaluation (will be removed during training)
    # and remove all other features for next period
    df_merged['price_nex'] = df_merged['price_next']
    df_merged['underlyingprice_nex'] = df_merged['underlyingprice_next']
    df_merged['optspread_nex'] = 2*(df_merged['ask_next']-df_merged['bid_next'])/(df_merged['bid_next']+
                                                                                  df_merged['ask_next'])
    df_merged['undspread_nex'] = (2*(df_merged['underlying_ask_next']-df_merged['underlying_bid_next'])/
                                  (df_merged['underlying_bid_next']+df_merged['underlying_ask_next']))
    df_merged['delta_nex'] = df_merged['delta_next']
    df = df_merged.loc[:, ~df_merged.columns.str.endswith('_next')].copy()

    #Adding of further features
    df.loc[:, 'moneyness'] = df['strike']/df['underlyingprice']
    df.loc[:, 'embedlev'] = np.abs(df['delta']) * (df['underlyingprice']/df['price'])
    df.loc[:, 'optspread'] = 2*(df['ask']-df['bid'])/(df['bid']+df['ask'])
    df.loc[:, 'undspread'] = 2*(df['underlying_ask']-df['underlying_bid'])/(df['underlying_bid']+df['underlying_ask'])
    df.loc[:, 'gamma'] = (df['gamma'] * df['underlyingprice'])/100 #Make gamma relative to stock price (see Bali et al.)
    df.loc[:, 'theta'] = (df['theta'] * df['underlyingprice'])/100 #Make theta relative to stock price (see Bali et al.)
    df.loc[:, 'vega'] = (df['vega'] * df['underlyingprice'])/100 #Make vega relative to stock price (see Bali et al.)

    if os.environ.get("DEBUG_SANITY") == "1":
        sanity_cols = ['moneyness', 'embedlev', 'optspread', 'undspread']
        nan_ratios = df[sanity_cols].isna().mean()
        print("DEBUG_SANITY NaN ratios:", nan_ratios.to_dict())
    df = df.drop(columns=['strike']) #Remove strike as we use moneyness as relative value

    #Convert categorical column to numeric using one-hot encoding
    df = pd.get_dummies(df, columns=['putcall'], drop_first=True, dtype=int)

    def interpolate_df2_into_df1(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        @brief Function that merges two dfs. However maturity is only available for fixed dates in one df.
            If those don't match use a linear interpolation of the features for the closest available maturities
        @param df1: Dataframe 1
        @param df2: Dataframe 2
        @return: Merged Dataframe
        """
        #Columns to interpolate (exclude keys)
        key_cols = ['loctimestamp', 'time', 'daystomaturity']
        value_cols = [col for col in df2.columns if col not in key_cols]

        out = df1.copy()
        out.loc[:, value_cols] = np.nan

        #Sort before interpolating (np.interp expects increasing x)
        df2_sorted = df2.sort_values(by=['loctimestamp', 'time', 'daystomaturity'], kind='mergesort')
        grouped_df2 = {
            k: g for k, g in df2_sorted.groupby(['loctimestamp', 'time'], sort=False)
        }

        #Batch interpolation per (loctimestamp, time)
        df1_groups = out.groupby(['loctimestamp', 'time'], sort=False).groups
        for (loctimestamp, time), idx in tqdm(df1_groups.items(), total=len(df1_groups)):
            df2_subset = grouped_df2.get((loctimestamp, time))
            if df2_subset is None:
                continue

            x = df2_subset['daystomaturity'].to_numpy()
            if x.size == 0:
                continue

            x_min = x.min()
            x_max = x.max()
            dtm = out.loc[idx, 'daystomaturity'].to_numpy()
            dtm_clip = np.clip(dtm, x_min, x_max)

            for col in value_cols:
                y = df2_subset[col].to_numpy()
                out.loc[idx, col] = np.interp(dtm_clip, x, y)

        if os.environ.get("DEBUG_CHECK") == "1":
            check_cols = value_cols[: min(20, len(value_cols))]
            if check_cols:
                sample_n = min(200, len(df1))
                sample_idx = df1.sample(n=sample_n, random_state=0).index

                slow = df1.loc[sample_idx].copy()
                slow.loc[:, check_cols] = np.nan
                for row_idx, row in slow.iterrows():
                    df2_subset = grouped_df2.get((row['loctimestamp'], row['time']))
                    if df2_subset is None:
                        continue
                    x = df2_subset['daystomaturity'].to_numpy()
                    if x.size == 0:
                        continue
                    x_min = x.min()
                    x_max = x.max()
                    dtm_clip = np.clip(row['daystomaturity'], x_min, x_max)
                    for col in check_cols:
                        y = df2_subset[col].to_numpy()
                        slow.at[row_idx, col] = np.interp(dtm_clip, x, y)

                np.testing.assert_allclose(
                    out.loc[sample_idx, check_cols].to_numpy(),
                    slow.loc[sample_idx, check_cols].to_numpy(),
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )
                print(f"DEBUG_CHECK passed (n={sample_n}, cols={len(check_cols)})")

        return out

    #Function to find the closest possible maturity
    def closest_maturity(original):
        return min(MATURITIES, key=lambda x: abs(x - original))

    #Merging of other dfs
    #As most of the other dfs only contain features for specific maturities find the closest
    # of theses to actual maturity
    df['possible_days'] = df['daystomaturity'].apply(closest_maturity)
    #Make sure loctimestamp is a datetime object
    df['loctimestamp'] = pd.to_datetime(df['loctimestamp'])
    df3['loctimestamp'] = pd.to_datetime(df3['loctimestamp'])
    df4['loctimestamp'] = pd.to_datetime(df4['loctimestamp'])
    df5['loctimestamp'] = pd.to_datetime(df5['loctimestamp'])
    df6['loctimestamp'] = pd.to_datetime(df6['loctimestamp'])
    #Split data into monthly dfs due to memory reasons
    monthly_dfs = {}
    for month in range(1, 13):
        monthly_dfs[month] = df[df['loctimestamp'].dt.month == month].copy()
    del df

    if not INTERPOLATE:
        #Rename maturity column in other dfs if not interpolation is to be done
        df3 = df3.rename(columns={'daystomaturity': 'possible_days'})
        df4 = df4.rename(columns={'daystomaturity': 'possible_days'})
        df6 = df6.rename(columns={'daystomaturity': 'possible_days'})
    else:
        #Merge other dfs first
        dfExtra = df3.merge(df4, on=['time', 'loctimestamp', 'daystomaturity'], how='inner')
        dfExtra = dfExtra.merge(df5, on=['time', 'loctimestamp'], how='inner')
        dfExtra = dfExtra.merge(df6, on=['time', 'loctimestamp', 'daystomaturity'], how='inner')

    sliced_items = list(monthly_dfs.items())
    sliced_dict = dict(sliced_items)
    #Merge month after month
    for month, df_month in sliced_dict.items():
        if month in [1,2,3,4,5,6,7,8,9,10,11,12]:
            print(f'Monat: {month}')
            if not INTERPOLATE:
                #If no interpolation is wished, just merge using closest available maturity
                df_month = df_month.merge(df3, on=['time', 'loctimestamp', 'possible_days'], how='inner')
                df_month = df_month.merge(df4, on=['time', 'loctimestamp', 'possible_days'], how='inner')
                df_month = df_month.merge(df5, on=['time', 'loctimestamp'], how='inner')
                df_month = df_month.merge(df6, on=['time', 'loctimestamp', 'possible_days'], how='inner')
            else:
                #Merge with interpolation
                df_month = interpolate_df2_into_df1(df_month,
                                                    dfExtra[dfExtra['loctimestamp'].dt.month == month])

            #Remove helper column
            df_month = df_month.drop(columns=['possible_days'])
            #Remove np.nan rows
            df_month = df_month.replace([np.inf, -np.inf], np.nan).dropna()
            df_month.reset_index(drop=True, inplace=True)

            #Save final monthly preprocessed dataset to parquet (compressed)
            df_month.to_parquet(
                OUT_DIR / f"data_month_{month}.parquet",
                engine="pyarrow",
                compression="zstd",
                index=False,
            )

            print('Summary')

            #Basic Info
            print("\n--- Basic Info ---")
            print(df_month.info())

            #Missing Values
            print("\n--- Missing Values ---")
            print(df_month.isnull().sum())

            #Summary of Numerical Columns
            print("\n--- Summary of Numerical Columns ---")
            print(df_month.describe())

            #Summary of Categorical Columns
            categorical_cols = df_month.select_dtypes(include=["object"]).columns
            if categorical_cols.any():
                print("\n--- Summary of Categorical Columns ---")
                for col in categorical_cols:
                    print(f"\nColumn: {col}")
                    print(df_month[col].value_counts().head(20))  #Show top 20 most common values

            #Check for Unique Values
            print("\n--- Unique Values Per Column ---")
            print(df_month.nunique())

            #Save a summary to xlsx
            summary = df_month.describe(include="all").transpose()
            summary["missing_values"] = df_month.isnull().sum()
            summary["unique_values"] = df_month.nunique()
            summary.to_excel(OUT_DIR / f"data_summary_{month}.xlsx")

            print("\nAnalysis complete! Summary saved as 'data_summary_{month}.xlsx'.")

if __name__ == "__main__":
    load_and_process_data()
