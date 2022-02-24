import time

import numpy as np
import pandas as pd
import pmdarima as arima
from arch import arch_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numba import jit, cuda, vectorize


def thousands(x, pos):
    return '${:,}'.format(int(float(x)))


def plot_histogram(series, ax):
    ax.hist(series, 50, facecolor='blue', alpha=0.5, density=True)
    ax.set_title('Profit/Loss Distribution', size=25)
    ax.set_xlabel('Profit/Loss', size=25)
    ax.set_ylabel('Distribution Density', size=25)
    ax.tick_params(labelrotation=45, labelsize=20)
    ax.xaxis.set_major_formatter(thousands)


def plot_series(simulated_series, last_actual_date, ax):
    for series in simulated_series:
        ax.plot(pd.date_range(last_actual_date, periods=len(series)).values, series)

    ax.set_title('Simulated Trajectories', size=25)
    ax.tick_params(labelrotation=45, labelsize=20)
    ax.set_xlabel('Time', size=25)
    ax.set_ylabel('Bitcoin Price', size=25)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.yaxis.set_major_formatter(thousands)


def var(results, risk=.05):
    results.sort()  # Sort them
    index = int(len(results) * risk)  # Count them and multiply by the risk factor
    return results[index]  # Return the value at that index


def Simulation_Statistics(results, simulated_arrays, last_observed_date):
    results = np.array(results)
    print('Average Profit/Loss: ${:,.2f}'.format(np.mean(results)))
    print('Profit/Loss Ranges from ${:,.2f} - ${:,.2f}'.format(np.min(results), np.max(results)))
    print('Probability of Earning a Return = {:.2f}%'.format(((results > 0).sum() / len(results)) * 100))
    print('The VaR at 95% Confidence is: ${:,.2f}'.format(var()))

    fig, axs = plt.subplots(1, 2, figsize=(13*1.10, 7*1.10))
    plot_histogram(results, axs[0])
    plot_series(simulated_arrays, last_observed_date, axs[1])

    fig.suptitle('Bitcoin Monte Carlo\nSimulating {} days'.format(len(simulated_arrays[0])), fontsize=30, fontweight='bold')
    fig.tight_layout()
    plt.show()


@cuda.jit
def one_step_ahead_arma_garch_volatility(series):
    arima_model_fitted = arima.auto_arima(series, information_criterion='bic')  # Fit an ARIMA model
    arima_residuals = arima_model_fitted.arima_res_.resid  # Retrieve the residuals
    model = arch_model(arima_residuals, vol='GARCH', p=1, q=1, rescale=False)  # Build Garch(1,1) model on ARIMA residuals
    fitted_model = model.fit(disp="off")  # Fit the model
    forecast = fitted_model.forecast(reindex=False)  # Forecast 1-step ahead

    return np.sqrt(forecast.residual_variance.values[0]) # Return 1-step ahead volatility


# def SimulateOnce(ts, trading_days):
#     for i in range(trading_days):
#         log_returns = np.diff(np.log(ts)) # Calculate log returns of the series
#         mean_return = log_returns.mean()  # Calculate the mean log return
#         volatility = one_step_ahead_arma_garch_volatility(log_returns)  # Calculate one-step ahead volatility
#         random_return = np.random.normal(  # Generate random return based on mean and volatility
#             (1 + mean_return) ** (1 / trading_days),
#             volatility / np.sqrt(trading_days), 1)
#         ts = np.append(ts, ts[-1] * random_return)  # Generate an estimated new price point given the random return
#
#     return ts[len(ts)-1:]



data = pd.read_csv('Bitcoin_2014-2022.csv', index_col=0)
data.index = pd.to_datetime(data.index)

series = np.array(data['Close'])
series = np.diff(np.log(series))


@vectorize(['int64(int64, int64)'], target='cuda')
def add_ufunc(x, y):
    return x + y



@cuda.jit
def add_kernel(x, y, out):
    start = cuda.grid(1)      # the 1 argument means a one dimensional thread grid, this returns a single value
    stride = cuda.gridsize(1) # ditto

    # assuming x and y inputs are same length
    for i in range(start, x.shape[0], stride):
        out[i] = x[i] + y[i]

n = 100000
x = np.arange(n).astype(np.float32)
y = 2 * x
out = np.zeros_like(x)

threads_per_block = 128
blocks_per_grid = 30

add_kernel[blocks_per_grid, threads_per_block](x, y, out)
# print(out)

one_step_ahead_arma_garch_volatility[blocks_per_grid, threads_per_block](series)

