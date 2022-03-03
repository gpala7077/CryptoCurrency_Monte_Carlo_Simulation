import numpy as np
import pandas as pd
import pmdarima
from arch import arch_model
from MonteCarlo_0 import MonteCarlo
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as stats
import yfinance as yf


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


def arma_garch_model(series, arima, arch_garch, show_warning=False):
    """
        Builds an ARMA-GARCH model and returns the model parameters

    Parameters
    ----------
    :param series:
        A numpy array containing the log return of a series

    :param arima:
        A dictionary containing the hyper-parameters for ARIMA model. For parameter information:
        https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html

    :param arch_garch
        A dictionary containing the hyper-parameters for the ARCH/GARCH process:
        https://arch.readthedocs.io/en/latest/univariate/introduction.html

    :param show_warning:
        A bool representing whether to show convergence warnings or not

    :return:
        Returns the forecast volatility
    """

    arima_model_fitted = pmdarima.auto_arima(series, **arima)  # Fit an ARIMA model
    arima_residuals = arima_model_fitted.arima_res_.resid  # Retrieve the residuals
    model = arch_model(arima_residuals, **arch_garch)  # Build Garch on ARMA residuals
    fitted_model = model.fit(disp="off", show_warning=show_warning)  # Fit the GARCH model

    return fitted_model


def forecast_sigma(model_parameters, returns, sigmas):
    """
        Forecasts the volatility given the models parameters, previous returns and volatilities
    :param model_parameters:
        A dictionary containing the model's parameters omega, alphas, and betas

    :param returns:
        A numpy array containing the previous returns

    :param sigmas:
        A numpy array containing the previous volatilities

    :return:
        Returns the new volatility
    """
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


def generate_random_return(mean_return, trading_days, volatility, risk_free_rate):
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

    :return:
        A float describing a random return
    """

    if risk_free_rate is None:
        # Method 1 to calculate random daily return based on the mean log-return and the volatility
        random_return = np.random.normal(  # Generate random return
            (1 + mean_return) ** (1 / trading_days), volatility / np.sqrt(trading_days), 1)
    else:
        # Method 2 to calculate random daily return using the risk-free rate and accounting for drag
        period_rate = (risk_free_rate - .5 * volatility ** 2) * (1 / trading_days)
        period_sigma = volatility * np.sqrt(1 / trading_days)
        random_return = np.random.normal(period_rate, period_sigma)

    return random_return


def SimulateGarch(ts, fitted_model, horizon, trading_days, risk_free_rate, arch_garch):
    """
        Generates a simulated series using an ARMA-GARCH process.
    Parameters
    ----------
    :param ts:
        The current time series

    :param fitted_model:
        The fitted ARMA-GARCH model

    :param horizon:
        An int describing the number of days to be forecasted

    :param trading_days:
        An int describing the total number of available trading days in a year

    :param risk_free_rate:
        A float describing the current risk-free rate

    :param arch_garch:
        The hyper-parameters of the GARCH model

    :return:
        Returns a numpy array containing the simulated series
    """

    actual = ts
    log_returns = np.diff(np.log(ts))
    sigmas = np.array([log_returns.std()]*arch_garch['q'])  # Calculate the current volatility
    random_return = np.array([log_returns[-max(arch_garch['p'], arch_garch['o']):]])  # Initialize an empty return array

    for i in range(horizon):
        random_return = np.append(  # Generate a new random return
            random_return, generate_random_return(0, trading_days, sigmas[-1], risk_free_rate))

        sigmas = np.append(  # Generate a new volatility
            sigmas, forecast_sigma(fitted_model.params, random_return[-max(arch_garch['p'], arch_garch['o']):],
                                   sigmas[-arch_garch['q']:]))

        # Generate a new price with the random return
        new_price = round(ts[-1] * random_return[-1], 2) if risk_free_rate is None else round(
            np.exp(np.log(ts[-1]) + random_return[-1]), 2)

        ts = np.append(ts, new_price)  # Append new price to the series

    simulated_series = ts[len(actual) - 1:]  # Store the simulated year array

    return simulated_series  # Return simulated series


def SimulateOptions(simulated_series, options_type, strike_price, num_interval=None):
    """
        Returns the payoff of an option
    Parameters
    ----------
    :param simulated_series:
        A numpy array that contains the stock price series

    :param options_type:
        A string input indicating the options type, i.e. Asian, European

    :param strike_price:
        A string or float input indicating the strike price, i.e. geometric, arithmetic, 54.65

    :param num_interval:
        An int that indicates the strike price interval. (Only used for Asian options)

    :return:
        Options payoff
    """

    if options_type == 'Asian':
        if num_interval is None:
            print('Asian options requires an interval period.')
            raise TypeError

        days = len(simulated_series)  # Number of days simulated
        days_interval = int(days / num_interval)  # Number of days in each interval to average end price
        price_lst = simulated_series[::days_interval]  # List of price intervals

    # Get average price based on method chosen
    if strike_price == 'arithmetic':
        strike_price = np.mean(price_lst)

    elif strike_price == 'geometric':
        strike_price = stats.gmean(price_lst)

    # Return the payoff
    return max(simulated_series[-1] - strike_price, 0)


class TimeSeries_MonteCarlo(MonteCarlo):
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
        self.options_info = options_info
        self.risk_free_rate = risk_free_rate
        self.ticker = yf.Ticker(ticker)
        self.arch_garch = dict(vol='GARCH', p=1, q=1, rescale=False,
                               dist='normal') if arch_garch is None else arch_garch
        self.arima = dict(information_criterion='bic') if arima is None else arima
        self.simulated_series = []
        self.results = []

        if model == 'Options' and options_info is None:
            print('Modeling options requires a dictionary')
            print("dict(type='Asian', strike='geometric', interval=4)")
            print("dict(type='European', strike=54.96, interval=None)")
            raise TypeError

        if isinstance(period, dict):
            if 'start' not in period or 'end' not in period:
                print('Period should have start and end dates in %Y-%m-%d format')
                raise TypeError
            self.ts = self.ticker.history(**period)
        else:
            self.ts = self.ticker.history(period=period)

        # Build ARMA-GARCH model on initialization
        self.fitted_model = arma_garch_model(
            np.diff(np.log(self.ts['Close'])), self.arima, self.arch_garch)

    def SimulateOnce(self):
        """ Simulate one price movement for the given horizon period"""

        simulated_series = SimulateGarch(self.ts['Close'], self.fitted_model, self.horizon, self.trading_days,
                                         self.risk_free_rate, self.arch_garch)

        self.simulated_series.append(simulated_series)

        if self.model == 'Returns':
            result = self.ts['Close'][-1] - simulated_series[-1]

        elif self.model == 'Options':
            result = SimulateOptions(simulated_series, self.options_info['type'], self.options_info['strike'],
                                     self.options_info['interval'])

        # Return result discounted by the risk-free rate, if no risk-free rate, then return result
        return result * np.exp(
            -self.risk_free_rate * (1 / self.trading_days)) if self.risk_free_rate is None else result

    def Simulation_Statistics(self):
        """Generates the relevant plots and statistics for the Monte Carlo simulation results"""

        self.results = np.array(self.results)
        vtime = self.ts.index[-1] - self.ts.index[0]
        years = int(vtime.days / 365)
        months = int((vtime.days % 365) / 30)
        days = int((vtime.days % 365) % 30)
        simulated_model = arma_garch_model(self.simulated_series[np.random.randint(0, len(self.simulated_series))],
                                           self.arima, self.arch_garch)

        print(self.elapsed_time)
        print('-' * len(self.elapsed_time))
        print('Simulated prices from {} year(s), {} month(s), and {} day(s) of historical data'.format(
            years, months, days))
        print('Average Profit/Loss: ${:,.2f}'.format(np.mean(self.results)))
        print('Profit/Loss Ranges from ${:,.2f} - ${:,.2f}'.format(np.min(self.results), np.max(self.results)))
        print('Probability of Earning a Return = {:.2f}%'.format(((self.results > 0).sum() / len(self.results)) * 100))
        print('The VaR at 95% Confidence is: ${:,.2f}'.format(self.var()))
        print('-' * len(self.elapsed_time))

        fig = plt.figure(constrained_layout=False, figsize=(18, 15))
        subplots = fig.subfigures(2, 2)

        ax0 = subplots[0, 0].subplots(1, 1)
        ax1 = subplots[0, 1].subplots(1, 1)
        ax2 = subplots[1, 0].subplots(2, 1)
        ax3 = subplots[1, 1].subplots(2, 1)

        plot_histogram(self.results, ax0)
        plot_series(self.simulated_series, self.ts.index[-1], ax1)
        plot_residuals_volatility(self.fitted_model, ax2)
        plot_residuals_volatility(simulated_model, ax3)
        subplots[1, 0].suptitle('Observed', y=.87, fontsize=30, fontweight='bold')
        subplots[1, 1].suptitle('Simulated', y=.87, fontsize=30, fontweight='bold')

        name = self.ticker.ticker if 'name' not in self.ticker.info else self.ticker.info['name']

        fig.suptitle('{} Monte Carlo\nRan {} Simulation(s) of {} day(s)'.format(
            name, self.sim_count, self.trading_days), fontsize=30, fontweight='bold')

        fig.subplots_adjust(hspace=.001)
        fig.tight_layout(pad=10)
        plt.show()
