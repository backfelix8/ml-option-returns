"""
@file further_graphics.py
@brief Plot further graphics

This file can be used to plot further graphics for the thesis.

@details
Values to set manually:
- COMBINATIONS - List of models used to plot R^2 and R^2 dispersion for (extended to two lists for two images)
- SIGNIFICANCE - Stars denoting the significance of the models in same order as in COMBINATIONS

Functions:
- plot_development_ai() - Create Figure 1.1 with data from https://arxiv.org/abs/2104.14337
- plot_gradient_descent() - Create Figure 3.4
- plot_results_models(df) - Create Figure 4.1
- plot_r_squared_dispersion(df) - Create Figure 4.2
- boxplot(df) - Creates a boxplot from given dataframe
"""

import os
os.environ["MPLBACKEND"] = "Agg"

from pathlib import Path

from plot_saver import save_current_figure

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

#Used models split in two groups (two figures) and significance stars
COMBINATIONS = [['GBR', 'RF', 'GBR-AV', 'RF-AV', 'Ens'],
                ['FFN', 'Main-SMF', 'Mod-SMF', 'JSMF', 'JSRF', 'Hyp', 'Att', 'AE']]
SIGNIFICANCE = [['***', ' ', '*', ' ', '***'], [' ', '*', ' ', ' ', '**', ' ', '*', ' ']]

#Some params for plotting
plt.rcParams["figure.figsize"] = (16,9)
plt.rcParams.update({'font.size': 22})

def plot_development_ai() -> None:
    """
    @brief Create Figure 1.1 with data from https://arxiv.org/abs/2104.14337
    """
    handwriting = [[1998, 2002, 2003, 2006, 2010, 2012, 2013, 2018],
                   [-100, -48, -26.67, -25.33, -20, -4, -1.33, 2.67]]
    speech = [[1998, 2011, 2013, 2014, 2015, 2016, 2017, 2018],
              [-100, -65.56, -52.7, -27.8, -8.71, -1.24, 0.41, 1.66]]
    image = [[2009, 2012, 2014, 2015, 2016, 2018, 2019, 2020],
             [-100, -44.16, -6.8, 0.69, 6.62, 11.69, 9.52, 16.45]]
    reading = [[2016, 2017, 2018, 2019, 2020], [-100, -8.89, 6.63, 18.12, 18.85]]
    language = [[2018, 2019, 2020, 2022],[-100, 3.73, 11.94, 15.67]]
    reasoning = [[2019, 2021, 2022, 2023], [-100, -80.54, -30.64, -0.62]]
    human = [[1998, 2023], [0, 0]]

    plt.plot(handwriting[0], handwriting[1], 'r-', label="Handwriting recognition")
    plt.plot(speech[0], speech[1], 'g-', label="Speech recognition")
    plt.plot(image[0], image[1], 'b-', label="Image recognition")
    plt.plot(reading[0], reading[1], 'y-', label="Reading comprehension")
    plt.plot(language[0], language[1], 'm-', label="Language understanding")
    plt.plot(reasoning[0], reasoning[1], 'c-', label="Predictive Reasoning")
    plt.plot(human[0], human[1], 'k-')
    plt.legend()
    save_current_figure(FIG_DIR, "development_ai")

def plot_gradient_descent() -> None:
    """
    @brief Create Figure 3.4
    """
    ax = plt.subplot(projection="3d", computed_zorder=False)
    step = 0.05
    x = -1
    y = -0.1
    data = [[-1,-0.1,2.2937]]
    for i in range(16): #Do 16 steps of gradient descent
        xNew = x - step * (8*x-8.4*x**3+2*x**5+y)
        yNew = y - step * (x-8*y+16*y**3)
        x = xNew
        y = yNew
        z = ((4-2.1*x**2+(x**4/3))*x**2+x*y+(-4+4*y**2)*y**2)
        data.append([xNew, yNew, z])
        x = xNew
        y = yNew

    #Plot the surface and scatter gradient descent points
    X = np.arange(-2, 2, 0.1)
    Y = np.arange(-1, 1, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = ((4-2.1*X**2+(X**4/3))*X**2+X*Y+(-4+4*Y**2)*Y**2)
    surf = ax.plot_surface(X, Y, Z, cmap="viridis",
                           linewidth=0, zorder=-1)
    ax.scatter(data[0][0], data[0][1], data[0][2], color="red", zorder=1)
    for el in data[1:]:
        ax.scatter(el[0], el[1], el[2], color="red", zorder=1)
    ax.set_zlim(-2, 6)
    save_current_figure(FIG_DIR, "gradient_descent")

def plot_results_models(df: pd.DataFrame) -> None:
    """
    @brief Create Figure 4.1
    @param df: result data from Results.xlsx
    """
    #Only select row containing R^2 values
    data = df.iloc[3]  # Series with MultiIndex: (Model, Put/Call/Both)

    for idx, el in enumerate(COMBINATIONS): #Create bars for each model
        data_used = data[el]

        #Convert to DataFrame with Models as rows and Put/Call/Both as columns
        df_restructured = data_used.unstack()  #Now DataFrame: rows = Model, cols = Put/Call/Both

        #Plotting params
        categories = ['Call', 'Put', 'All']
        colors = ['lightsteelblue', 'slategray', 'black']
        models = df_restructured.index.tolist()
        x = np.arange(len(models))
        width = 0.25

        fig, ax = plt.subplots()
        #Plot bars for only Put, only Call and Both
        for i, (cat, color) in enumerate(zip(categories, colors)):
            ax.bar(x + (i - 1) * width, df_restructured[cat], width=width, label=cat, color=color)

        #Customize plot
        ax.axhline(0, color='black', linewidth=1, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_ylabel('$R^2$')

        #Set significance asterisks
        asterisks = SIGNIFICANCE[idx]
        for i, symbol in enumerate(asterisks[:len(models)]):
            if symbol.strip():
                ax.text(x[i], -0.01, symbol, ha='center', va='top', fontsize=22)

        plt.tight_layout()
        save_current_figure(FIG_DIR, f"results_models_{idx}")

def plot_r_squared_dispersion(df: pd.DataFrame) -> None:
    """
    @brief Create Figure 4.2
    @param df: result data from Results.xlsx
    """
    data = df.iloc[7:22, :] #Get R^2 for individual runs
    data = data.loc[:, df.columns.get_level_values(1) == "All"] #Only take values from Calls and Puts
    data.columns = [col[0] for col in data.columns]
    boxplot(data) #Create boxplot with R^2 dispersion

def boxplot(df: pd.DataFrame) -> None:
    """
    @brief Creates a boxplot from given dataframe
    @param df: data
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
    colors = sns.color_palette("Spectral", len(df.columns))
    for i in range(len(df.columns)): #Iterate through each model
        scores = df.iloc[:, i] #Take data for i-th model only
        scores = scores.dropna()
        stats = custom_boxplot_stats(scores)

        #Draw plot
        plt.plot([i, i], [stats['whislo'], stats['q1']], color='k')  #lower whisker
        plt.plot([i - 0.2, i + 0.2], [stats['whislo'], stats['whislo']], color='k',
                 linewidth=1.2) #lower whisker
        plt.plot([i, i], [stats['whishi'], stats['q3']], color='k')  #upper whisker
        plt.plot([i - 0.2, i + 0.2], [stats['whishi'], stats['whishi']], color='k',
                 linewidth=1.2) #upper whisker
        plt.fill_betweenx(
            [stats['q1'], stats['q3']], i - 0.4, i + 0.4,
            color=colors[i], edgecolor='k', linewidth=1.2
        ) #colored box
        # median
        plt.plot([i - 0.4, i + 0.4], [stats['med'], stats['med']], color='k', linewidth=1.2)
        # mean
        plt.plot(i, np.mean(scores), 'o', color='white', markersize=6, markeredgecolor='k')

    #Customize plot
    plt.xticks(range(len(df.columns)), df.columns, rotation=45)
    plt.axhline(0, linestyle='--', color='black', linewidth=0.8)
    plt.ylabel("$R^2$")
    plt.tight_layout()
    save_current_figure(FIG_DIR, "r_squared_dispersion_boxplot")

if __name__ == '__main__':
    # Load results from Results.xlsx
    df = pd.read_excel("../analysis/results.xlsx", index_col=0)
    # Convert the columns to MultiIndex (assumes format like 'FFN All')
    multi_cols = [tuple(col.split()) for col in df.columns]
    df.columns = pd.MultiIndex.from_tuples(multi_cols)

    #Create plots
    plot_development_ai()
    plot_gradient_descent()
    plot_results_models(df)
    plot_r_squared_dispersion(df)
