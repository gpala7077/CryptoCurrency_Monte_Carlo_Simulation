import numpy as np
import pandas as pd
import pmdarima as arima
from arch import arch_model
from MonteCarlo_0 import MonteCarlo
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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


class TimeSeries_MonteCarlo(MonteCarlo):

    def __init__(self, ts, trading_days=365):
        self.ts = ts
        self.trading_days = trading_days
        self.simulated_arrays = []
        self.results = []

    def one_step_ahead_arma_garch_volatility(self, series):
        arima_model_fitted = arima.auto_arima(series, information_criterion='bic')  # Fit an ARIMA model
        arima_residuals = arima_model_fitted.arima_res_.resid  # Retrieve the residuals
        model = arch_model(arima_residuals, vol='GARCH', p=1, q=1, rescale=False)  # Build Garch(1,1) model on ARIMA residuals
        fitted_model = model.fit(disp="off")  # Fit the model
        forecast = fitted_model.forecast(reindex=False)  # Forecast 1-step ahead

        return np.sqrt(forecast.residual_variance.values[0]) # Return 1-step ahead volatility

    def SimulateOnce(self):
        ts = self.ts['Close']
        for i in range(self.trading_days):
            log_returns = np.diff(np.log(ts)) # Calculate log returns of the series
            mean_return = log_returns.mean()  # Calculate the mean log return
            volatility = self.one_step_ahead_arma_garch_volatility(log_returns)  # Calculate one-step ahead volatility
            random_return = np.random.normal(  # Generate random return based on mean and volatility
                (1 + mean_return) ** (1 / self.trading_days),
                volatility / np.sqrt(self.trading_days), 1)
            ts = np.append(ts, ts[-1] * random_return)  # Generate an estimated new price point given the random return

        self.simulated_arrays.append(ts[len(self.ts)-1:])  # Store the simulated year array
        return ts[-1] - self.ts['Close'][-1]  # Return the ending price point less the last actual observed price

    def Simulation_Statistics(self):
        self.results = np.array(self.results)
        print('Average Profit/Loss: ${:,.2f}'.format(np.mean(self.results)))
        print('Profit/Loss Ranges from ${:,.2f} - ${:,.2f}'.format(np.min(self.results), np.max(self.results)))
        print('Probability of Earning a Return = {:.2f}%'.format(((self.results > 0).sum() / len(self.results)) * 100))
        print('The VaR at 95% Confidence is: ${:,.2f}'.format(self.var()))

        fig, axs = plt.subplots(1, 2, figsize=(13*1.10, 7*1.10))
        plot_histogram(self.results, axs[0])
        plot_series(self.simulated_arrays, self.ts.index[-1], axs[1])

        fig.suptitle('Bitcoin Monte Carlo\nSimulating {} days'.format(self.trading_days), fontsize=30, fontweight='bold')
        fig.tight_layout()
        plt.show()


# download dataframe
data = pd.read_csv('Bitcoin_2014-2022.csv', index_col=0)
data.index = pd.to_datetime(data.index)

TS = TimeSeries_MonteCarlo(ts=data, trading_days=365)
TS.RunSimulation(50)
TS.Simulation_Statistics()


