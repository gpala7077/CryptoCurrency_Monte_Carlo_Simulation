import numpy as np
import pandas as pd
import pmdarima as arima
from arch import arch_model
from MonteCarlo_0 import MonteCarlo
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def plot_histogram(series, ax):
    ax.hist(series, 50, facecolor='blue', alpha=0.5, density=True)
    ax.set_title('Final Price Point', fontweight="bold", size=30)


def plot_series(simulated_series, last_actual_date, ax):
    for series in simulated_series:
        ax.plot(pd.date_range(last_actual_date, periods=len(series)).values, series)

    ax.set_title('Simulated Trajectories', fontweight="bold", size=30)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))


class TimeSeries_MonteCarlo(MonteCarlo):

    def __init__(self, ts, trading_days=365):
        self.ts = ts
        self.trading_days = trading_days
        self.simulated_arrays = []

    def one_step_ahead_arma_garch_volatility(self, series):
        arima_model_fitted = arima.auto_arima(series, information_criterion='bic')  # Fit an ARIMA model
        arima_residuals = arima_model_fitted.arima_res_.resid  # Retrieve the residuals
        model = arch_model(arima_residuals, vol='GARCH', p=1, q=1,rescale=False)  # Build Garch(1,1) model on ARIMA residuals
        fitted_model = model.fit(disp="off")  # Fit the model
        forecast = fitted_model.forecast(reindex=False)  # Forecast 1-step ahead

        return np.sqrt(forecast.residual_variance.values[0]) # Return 1-step ahead volatility

    def SimulateOnce(self):
        ts = self.ts['Close']
        for i in range(self.trading_days):
            log_returns = np.diff(np.log(ts))  # Calculate log returns of the series
            mean_return = log_returns.mean()  # Calculate the mean log return
            volatility = self.one_step_ahead_arma_garch_volatility(log_returns)  # Calculate one-step ahead volatility
            random_return = np.random.normal(  # Generate random return based on mean and volatility
                (1 + mean_return) ** (1 / self.trading_days),
                volatility / np.sqrt(self.trading_days), 1)
            ts = np.append(ts, ts[-1] * random_return)  # Generate an estimated new price point given the random return

        self.simulated_arrays.append(ts[len(self.ts)-1:])  # Store the simulated year array
        return ts[-1]  # Return the ending price point

    def Simulation_Statistics(self):
        print('Average Ending Price: ${:,.2f}'.format(np.mean(self.results)))
        print('Ending Price Ranges from ${:,.2f} - ${:,.2f}'.format(np.min(self.results), np.max(self.results)))
        print('The VaR at 95% Confidence is: ${:,.2f}'.format(self.var()))

        fig, axs = plt.subplots(1, 2, figsize=(15, 10))
        plot_histogram(self.results, axs[0])
        plot_series(self.simulated_arrays, self.ts.index[-1], axs[1])
        plt.xticks(rotation=45)
        plt.show()


# download dataframe
data = pd.read_csv('Bitcoin_2014-2022.csv', index_col=0)
TS = TimeSeries_MonteCarlo(ts=data, trading_days=365)
TS.RunSimulation(5)
TS.Simulation_Statistics()

