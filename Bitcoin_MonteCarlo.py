import datetime

import numpy as np
import pandas as pd
import pmdarima
from arch import arch_model
from MonteCarlo_0 import MonteCarlo
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as stats
import yfinance as yf
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


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


def generate_random_return(mean_return, trading_days, volatility, risk_free_rate, generate=1):
    """
        Generate a random return either using the mean log return and volatility or the risk-free rate and volatility

    :param mean_return:
        A float describing the mean of the log-returns

    :param trading_days:
        An int describing the number of available trading days. i.e. Stocks = 253, Crypto=365

    :param volatility:
        A float describing the volatility for the given financial instrument

    :param risk_free_rate:
        A float describing the current risk-free rate

    :param generate:
        An int describing the number of random returns to generate

    :return:
        A float describing a random return
    """

    if risk_free_rate is None:
        # Method 1 to calculate random daily return based on the mean log-return and the volatility
        random_return = np.random.normal(  # Generate random return
            (1 + mean_return) ** (1 / trading_days), volatility / np.sqrt(trading_days), generate)
    else:
        # Method 2 to calculate random daily return using the risk-free rate and accounting for drag
        period_rate = (risk_free_rate - .5 * volatility ** 2) * (1 / trading_days)
        period_sigma = volatility * np.sqrt(1 / trading_days)
        random_return = np.random.normal(period_rate, period_sigma, generate)

    return random_return


class Arma_Garch_modeler:
    def __init__(self, ts, arima=None, arch_garch=None, show_warnings=False):
        """
             A Class that fits an ARMA-GARCH model

        Parameters
        ----------
        :param ts:
            A timeseries dataset to build the ARMA-GARCH model

        :param arima:
            A dictionary containing the hyper-parameters for ARIMA model. For parameter information:
            https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html

        :param arch_garch
            A dictionary containing the hyper-parameters for the ARCH/GARCH process:
            https://arch.readthedocs.io/en/latest/univariate/introduction.html

        :param show_warning:
            A bool representing whether to show convergence warnings or not
"""
        self.arch_garch = dict(vol='GARCH', p=1, q=1, o=0, mean="Zero",
                               rescale=True, dist='normal') if arch_garch is None else arch_garch

        self.arima = dict(information_criterion='bic') if arima is None else arima
        self.show_warnings = show_warnings

        # Build ARMA-GARCH model on initialization
        if not {'vol', 'p', 'q', 'o', 'mean', 'rescale', 'dist'} <= self.arch_garch.keys():
            print('GARCH parameters must at minimum include the following parameters')
            print('dict(vol="GARCH", p=1, q=1, o=0, mean="Zero", rescale=True, dist="normal")')
            raise TypeError

        self.fitted_model = self.arma_garch_model(ts)

    def arma_garch_model(self, series):
        """
            Builds an ARMA-GARCH model and returns the model parameters

        Parameters
        ----------
        :param series:
            A numpy array containing the log return of a series

        :return:
            Fitted ARMA-GARCH model
        """
        arima_model_fitted = pmdarima.auto_arima(series, **self.arima)  # Fit an ARIMA model
        arima_residuals = arima_model_fitted.arima_res_.resid  # Retrieve the residuals
        model = arch_model(arima_residuals, **self.arch_garch)  # Build Garch on ARMA residuals
        fitted_model = model.fit(disp="off", show_warning=self.show_warnings)  # Fit the GARCH model
        return fitted_model

    def forecast_sigma(self, returns, sigmas):
        """
            Forecasts the volatility given the models parameters, previous returns and volatilities

        :param returns:
            A numpy array containing the previous returns

        :param sigmas:
            A numpy array containing the previous volatilities

        :return:
            Returns the new volatility
        """
        model_parameters = self.fitted_model.params
        omega = model_parameters['omega']
        alphas = np.array(model_parameters[[alpha for alpha in model_parameters.keys() if 'alpha' in alpha]])
        betas = np.array(model_parameters[[beta for beta in model_parameters.keys() if 'beta' in beta]])
        gammas = np.array(model_parameters[[gamma for gamma in model_parameters.keys() if 'gamma' in gamma]])

        if len(alphas) > 0 and len(betas) > 0 and len(gammas) > 0:
            return np.sqrt(omega + np.sum(alphas * (returns[-len(alphas):] ** 2)) +
                           np.sum(gammas * (returns[-len(gammas):] ** 2) * (returns[-len(gammas):] < 0)) +
                           np.sum(betas * (sigmas ** 2)))

        elif len(alphas) > 0 and len(betas) > 0:
            return np.sqrt(omega + np.sum(alphas * (returns ** 2)) + np.sum(betas * (sigmas ** 2)))

        elif len(alphas) > 0:
            return np.sqrt(omega + np.sum(alphas * (returns ** 2)))

        elif len(betas) > 0:
            return np.sqrt(omega + np.sum(betas * (sigmas ** 2)))

    def simulate_garch(self, ts, horizon, trading_days, risk_free_rate):
        """
            Generates a simulated series using an ARMA-GARCH process.
        Parameters
        ----------
        :param ts:
            The current time series

        :param horizon:
            An int describing the number of days to be forecasted

        :param trading_days:
            An int describing the total number of available trading days in a year

        :param risk_free_rate:
            A float describing the current risk-free rate

        :return:
            Returns a numpy array containing the simulated series
        """

        actual = ts
        log_returns = np.diff(np.log(ts))
        sigmas = np.array([log_returns.std()] * self.arch_garch['q'])  # Calculate the current volatility
        random_return = np.array(generate_random_return(0, trading_days, sigmas[-1], risk_free_rate,
                                                        generate=max(self.arch_garch['p'], self.arch_garch['o'])))
        for i in range(horizon):
            random_return = np.append(  # Generate a new random return
                random_return, generate_random_return(0, trading_days, sigmas[-1], risk_free_rate))

            sigmas = np.append(  # Generate a new volatility
                sigmas, self.forecast_sigma(
                    random_return[-max(self.arch_garch['p'], self.arch_garch['o']):], sigmas[-self.arch_garch['q']:]))

            # Generate a new price with the random return
            new_price = round(ts[-1] * random_return[-1], 2) if risk_free_rate is None else round(
                np.exp(np.log(ts[-1]) + random_return[-1]), 2)

            ts = np.append(ts, new_price)  # Append new price to the series

        simulated_series = ts[len(actual) - 1:]  # Store the simulated year array

        return simulated_series  # Return simulated series


class Financial_Timeseries(yf.Ticker):
    valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']

    def __init__(self, ticker, period='max'):
        super().__init__(ticker)

        if isinstance(period, dict) and not {'start', 'end'} == period.keys():
            print('Period should be a dict(start="2000-01-01",end="2002-01-01"')
            print('Period should have start and end dates in %Y-%m-%d format')
            raise TypeError

        elif isinstance(period, str) and period not in self.valid_periods:
            print('Period should be a valid time period:')
            print(valid_periods)
            raise TypeError

        elif isinstance(period, dict):
            try:
                self.timeseries = self.history(**period)
            except ValueError:
                print('Period should be in %Y-%m-%d format')
                raise ValueError

        elif isinstance(period, str):
            self.timeseries = self.history(period=period)

    def plot_timeseries(self, series=None, fig_size=(10, 10)):

        fig, axs = plt.subplots(4, 1, figsize=fig_size, squeeze=True)
        fig.suptitle('{} Timeseries of {}'.format(series, self.info['name'] if 'name' in self.info else self.ticker))

        series = ['Open', 'High', 'Low', 'Close'] if series is None else series
        values = self.timeseries[series]
        returns = self.transform(series, 'returns')
        log_returns = self.transform(series, 'log returns')

        sns.lineplot(data=values, ax=axs[0])
        sns.lineplot(data=returns, ax=axs[1])
        sns.lineplot(data=log_returns, ax=axs[2])
        axs[3].bar(x=self.timeseries.index, height=self.timeseries['Volume'])

        axs[0].title.set_text('Values')
        axs[0].tick_params(labelrotation=45)

        axs[1].title.set_text('Returns')
        axs[1].tick_params(labelrotation=45)

        axs[2].title.set_text('Log Returns')
        axs[2].tick_params(labelrotation=45)

        axs[3].title.set_text('Daily Volume')
        axs[3].tick_params(labelrotation=45)

        fig.tight_layout()

    def plot_ACF_PACF(self, series='Close', transform=None):

        fig, axs = plt.subplots(4, 1, figsize=(10, 10), squeeze=True)
        fig.suptitle('{} AutoCorrelation Plots of the {}'.format(series, 'Values' if transform is None else transform))

        series = self.timeseries[series] if transform is None else self.transform(series, transform)

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

    def transform(self, series=None, transform='log returns'):

        series = ['Open', 'High', 'Low', 'Close'] if series is None else series

        if transform == 'returns':
            transform = self.timeseries.pct_change()

        elif transform == 'log returns':
            transform = np.log(1 + self.timeseries.pct_change())

        return transform[series].dropna()

    def __str__(self):
        vtime = self.timeseries.index[-1] - self.timeseries.index[0]
        years = int(vtime.days / 365)
        months = int((vtime.days % 365) / 30)
        days = int((vtime.days % 365) % 30)
        return 'Reviewing {} year(s), {} month(s), and {} day(s) of historical data'.format(years, months, days)


class Option:
    def __init__(self, options_type, strike_price, call, contract_price, interval=None):

        if options_type not in ['Asian', 'European']:
            print('Can only model Asian or European options')
            raise TypeError

        if options_type == 'Asian' and \
                (not isinstance(strike_price, str) or strike_price not in ['geometric', 'arithmetic'] or
                 not interval > 0 or interval is None):

            print('Only Asian options can have a mean strike price, geometric or arithmetic.\n'
                  'They must also have an interval greater than 0')
            raise TypeError

        elif options_type != 'Asian' and (not isinstance(strike_price, float) or not isinstance(strike_price, int)):
            print('Strike price needs to be a float or int')
            raise TypeError

        if not isinstance(call, bool):
            print('Call needs to be a boolean, True (for Call) False (for Put)')
            raise TypeError
        self.options_type = options_type
        self.strike_price = strike_price
        self.contract_price = contract_price
        self.interval = interval
        self.call = call

    def simulate_options(self, simulated_series):
        """
            Returns the payoff of an option
        Parameters
        ----------
        :param simulated_series:
            A numpy array describing the prices

        :return:
            Options payoff
        """

        if self.options_type == 'Asian':
            days = len(simulated_series)  # Number of days simulated
            days_interval = int(days / self.interval)  # Number of days in each interval to average end price
            price_lst = simulated_series[::days_interval]  # List of price intervals

            # Get average price based on method chosen
            if self.strike_price == 'arithmetic':
                strike_price = np.mean(price_lst)

            elif self.strike_price == 'geometric':
                strike_price = stats.gmean(price_lst)

        # Return the payoff
        return max(simulated_series[-1] - strike_price - self.contract_price, -self.contract_price) if self.call \
            else max(strike_price - simulated_series[-1] - self.contract_price, -self.contract_price)


class Timeseries_MonteCarlo(MonteCarlo):
    """
        A Monte Carlo class that simulates the price movement of a stock using an ARMA-GARCH process.

    Attributes
    ----------
    ticker: str
        The ticker for the requested stock

    period: str or dict
        Period of time for the stock. i.e. max, 1mo, 1d, ytd, or dict(start=2014-01-01,end=2017-04-03)

    model: str
        Type of return to be evaluated. i.e. Returns or Options

    horizon: int
        Number of days to forecast

    trading_days: int
        Number of available trading days. stocks = 253, cryptocurrencies = 365

    risk_free_rate: float
        The current risk-free rate

    options_info: dict
        Option information. i.e.
        dict(type='Asian', strike='geometric', interval=4) or dict(type='European', strike=54.96, interval=None)

    arima: dict
        A dictionary containing the hyper-parameters for an ARIMA model

    arch_garch: dict
        A dictionary containing the hyper-parameters for an ARCH/GARCH model

    """

    def __init__(self, ticker, period='max', model='Returns', horizon=365, trading_days=365,
                 risk_free_rate=.03, options_info=None, arima=None, arch_garch=None):

        self.trading_days = trading_days
        self.horizon = horizon
        self.model = model
        self.options = Option(**options_info) if options_info is not None else options_info
        self.risk_free_rate = risk_free_rate
        self.data = Financial_Timeseries(ticker, period)
        self.arma_garch = Arma_Garch_modeler(self.data.transform('Close', 'log returns'), arima, arch_garch)
        self.simulated_series = []
        self.results = []

    def simulate_once(self):
        """ Simulate one price movement for the given horizon period"""

        simulated_series = self.arma_garch.simulate_garch(self.data.timeseries['Close'], self.horizon,
                                                          self.trading_days, self.risk_free_rate)

        self.simulated_series.append(simulated_series)

        if self.model == 'Returns':
            result = self.data.timeseries[-1] - simulated_series[-1]

        elif self.model == 'Options':
            result = self.options.simulate_options(simulated_series)

        # Return result discounted by the risk-free rate, if no risk-free rate, then return result
        return result * np.exp(
            -self.risk_free_rate * (1 / self.trading_days)) if self.risk_free_rate is None else result

    def simulation_statistics(self, risk=.05):
        """
            Generates the relevant plots and statistics for the Monte Carlo simulation results

        :param risk:
            A float with the range between 0 and 1, indicating the value at risk level
        :return:
            None
        """

        self.results = np.array(self.results)
        simulated_model = self.arma_garch.arma_garch_model(
            np.diff(np.log(self.simulated_series[np.random.randint(0, len(self.simulated_series))])))

        print(self.elapsed_time)
        print('-' * len(self.elapsed_time))
        print(self.data)
        print('Average Profit/Loss: ${:,.2f}'.format(np.mean(self.results)))
        print('Profit/Loss Ranges from ${:,.2f} - ${:,.2f}'.format(np.min(self.results), np.max(self.results)))
        print('Probability of Earning a Return = {:.2f}%'.format(((self.results > 0).sum() / len(self.results)) * 100))
        print('The VaR at 95% Confidence is: ${:,.2f}'.format(self.var(risk)))
        print('-' * len(self.elapsed_time))

        fig = plt.figure(constrained_layout=False, figsize=(18, 15))
        subplots = fig.subfigures(2, 2)

        ax0 = subplots[0, 0].subplots(1, 1)
        ax1 = subplots[0, 1].subplots(1, 1)
        ax2 = subplots[1, 0].subplots(2, 1)
        ax3 = subplots[1, 1].subplots(2, 1)

        plot_histogram(self.results, ax0)
        plot_series(self.simulated_series, self.data.timeseries.index[-1], ax1)
        plot_residuals_volatility(self.arma_garch.fitted_model, ax2)
        plot_residuals_volatility(simulated_model, ax3)
        subplots[1, 0].suptitle('Observed', y=.87, fontsize=30, fontweight='bold')
        subplots[1, 1].suptitle('Simulated', y=.87, fontsize=30, fontweight='bold')

        name = self.data.ticker if 'name' not in self.data.info else self.data.info['name']

        fig.suptitle('{} Monte Carlo\nRan {} Simulation(s) of {} day(s)'.format(
            name, self.sim_count, self.trading_days), fontsize=30, fontweight='bold')

        fig.subplots_adjust(hspace=.001)
        fig.tight_layout(pad=10)
        plt.show()
