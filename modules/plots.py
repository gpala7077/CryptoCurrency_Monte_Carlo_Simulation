import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd


def thousands(x, pos):
    """
        Formats a string in comma format. Used in ax.xaxis.set_major_formatter or ax.yaxis.set_major_formatter

    :param x:
        String containing a float

    :param pos:
        Position in matplotlib

    :return:
        None
    """

    return '${:,}'.format(int(float(x)))


def plot_histogram(series, ax):
    """
        Plots a histogram

    Parameters
    ----------
    :param series:
        A numpy array

    :param ax:
        Matplotlib ax object

    :return:
        None
    """

    ax.hist(series, 50, facecolor='blue', alpha=0.5, density=True)
    ax.set_title('Profit/Loss Distribution', size=25)
    ax.set_xlabel('Profit/Loss', size=25)
    ax.set_ylabel('Distribution Density', size=25)
    ax.tick_params(labelrotation=45, labelsize=20)
    ax.xaxis.set_major_formatter(thousands)


def plot_series(simulated_series, last_actual_date, ax):
    """
        Plots all the simulated time series

    Parameters
    ----------
    :param simulated_series:
        A list containing numpy arrays

    :param last_actual_date:
        Last observed date

    :param ax:
        Matplotlib ax object

    :return:
        None
    """

    for series in simulated_series:
        ax.plot(pd.date_range(last_actual_date, periods=len(series)).values, series)

    ax.set_title('Simulated Trajectories', size=25)
    ax.tick_params(labelrotation=45, labelsize=20)
    ax.set_xlabel('Time', size=25)
    ax.set_ylabel('Bitcoin Price', size=25)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.yaxis.set_major_formatter(thousands)


def plot_residuals_volatility(model, ax):
    ax[0].plot(model.resid / model.conditional_volatility)
    ax[1].plot(model.conditional_volatility)

    ax[0].set_title('Standardized Residuals', size=25)
    ax[1].set_title('Conditional Volatility', size=25)


def plot_montecarlo_simulation_results(f_timeseries, results, simulated_series, observed_model, simulated_model):
    fig = plt.figure(constrained_layout=False, figsize=(18, 15))
    subplots = fig.subfigures(2, 2)

    ax0 = subplots[0, 0].subplots(1, 1)
    ax1 = subplots[0, 1].subplots(1, 1)
    ax2 = subplots[1, 0].subplots(2, 1)
    ax3 = subplots[1, 1].subplots(2, 1)

    plot_histogram(results, ax0)
    plot_series(simulated_series, f_timeseries.timeseries.index[-1], ax1)
    plot_residuals_volatility(observed_model, ax2)
    plot_residuals_volatility(simulated_model, ax3)
    subplots[1, 0].suptitle('Observed', y=.87, fontsize=30, fontweight='bold')
    subplots[1, 1].suptitle('Simulated', y=.87, fontsize=30, fontweight='bold')

    name = f_timeseries.ticker if 'name' not in f_timeseries.info else f_timeseries.info['name']

    fig.suptitle('{} Monte Carlo\nRan {:,} Simulation(s) of {:,} day(s)'.format(
        name, len(results), simulated_series[0].size-1), fontsize=30, fontweight='bold')

    fig.subplots_adjust(hspace=.001)
    fig.tight_layout(pad=10)
    plt.show()


def plot_timeseries(f_timeseries, series=None, fig_size=(10, 10)):
    fig, axs = plt.subplots(4, 1, figsize=fig_size, squeeze=True)
    fig.suptitle('{} Timeseries'.format(
        f_timeseries.info['name'] if 'name' in f_timeseries.info else f_timeseries.ticker))

    series = ['Open', 'High', 'Low', 'Close'] if series is None else series
    values = f_timeseries.timeseries[series]
    returns = f_timeseries.transform(series, 'returns')
    log_returns = f_timeseries.transform(series, 'log returns')

    sns.lineplot(data=values, ax=axs[0])
    sns.lineplot(data=returns, ax=axs[1])
    sns.lineplot(data=log_returns, ax=axs[2])
    axs[3].bar(x=f_timeseries.timeseries.index, height=f_timeseries.timeseries['Volume'])

    axs[0].title.set_text('Values')
    axs[0].tick_params(labelrotation=45)

    axs[1].title.set_text('Returns')
    axs[1].tick_params(labelrotation=45)

    axs[2].title.set_text('Log Returns')
    axs[2].tick_params(labelrotation=45)

    axs[3].title.set_text('Daily Volume')
    axs[3].tick_params(labelrotation=45)

    fig.tight_layout()


def plot_ACF_PACF(f_timeseries, series='Close', transform=None):
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), squeeze=True)
    fig.suptitle('{} AutoCorrelation Plots of the {}'.format(series, 'Values' if transform is None else transform))

    series = f_timeseries.timeseries[series] if transform is None else f_timeseries.transform(series, transform)

    plot_acf(series, ax=axs[0], zero=False)
    plot_pacf(series, ax=axs[1], zero=False)
    plot_acf(series ** 2, ax=axs[2], zero=False)
    plot_acf(abs(series), ax=axs[3], zero=False)

    axs[0].title.set_text('ACF')
    axs[0].tick_params(labelrotation=45)

    axs[1].title.set_text('PACF')
    axs[1].tick_params(labelrotation=45)

    axs[2].title.set_text('ACF squared')
    axs[2].tick_params(labelrotation=45)

    axs[3].title.set_text('ACF Absolute value')
    axs[3].tick_params(labelrotation=45)

    fig.tight_layout()
