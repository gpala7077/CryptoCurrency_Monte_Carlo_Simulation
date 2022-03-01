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


def arma_garch_volatility(series, horizon, arima, arch_garch):
    """
        Builds an ARMA-GARCH model and returns the forecasted volatility
    Parameters
    ----------
    :param series:
        A numpy array containing the log return of a series

    :param horizon:
        An int describing the number of days to be forecasted

    :param arima:
        A dictionary containing the hyper-parameters for ARIMA model. For parameter information:
        https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html

    :param arch_garch
        A dictionary containing the hyper-parameters for the ARCH/GARCH process:
        https://arch.readthedocs.io/en/latest/univariate/introduction.html

    :return:
        Returns the forecast volatility
    """

    arima_model_fitted = pmdarima.auto_arima(series, **arima)  # Fit an ARIMA model
    arima_residuals = arima_model_fitted.arima_res_.resid  # Retrieve the residuals
    model = arch_model(arima_residuals, **arch_garch)  # Build Garch on ARMA residuals
    fitted_model = model.fit(disp="off")  # Fit the model
    forecast = fitted_model.forecast(horizon=horizon, reindex=False)  # Forecast n-step ahead

    return np.sqrt(forecast.residual_variance.values[0])  # Return volatility forecast


def SimulateGarch(ts, horizon, trading_days, rebuild_rate, risk_free_rate, arima, arch_garch):
    """
        Generates a simulated series using an ARMA-GARCH process.
    Parameters
    ----------
    :param ts:
        A numpy array containing the observed price values of a stock

    :param horizon:
        An int describing the number of days to be forecasted

    :param trading_days:
        An int describing the total number of available trading days in a year

    :param rebuild_rate:
        An int describing the rate at which the ARMA-GARCH model is rebuilt

    :param risk_free_rate:
        A float describing the current risk-free rate

    :param arima:
        A dictionary containing the hyper-parameters for an ARIMA model

    :param arch_garch
        A dictionary containing the hyper-parameters for an ARCH/GARCH model

    :return:
        Returns a numpy array containing the simulated series
    """
    actual = ts
    n_steps = 0
    volatility = []
    for i in range(horizon):
        log_returns = np.diff(np.log(ts))  # Calculate log returns of the series
        mean_return = log_returns.mean()  # Calculate the mean log return
        log_price = np.log(ts)

        if n_steps == len(volatility):
            volatility = arma_garch_volatility(log_returns, rebuild_rate, arima, arch_garch)  # Calculate volatility
            n_steps = 0

        # Method 1 to calculate random return
        # random_return = np.random.normal(  # Generate random return
        #     (1 + mean_return) ** (1 / trading_days), volatility[n_steps] / np.sqrt(trading_days), 1)
        #
        # ts = np.append(ts, ts[-1] * random_return)  # Generate an estimated new price with the random return

        # Method 2 to calculate random return
        period_rate = (risk_free_rate - .5 * volatility[n_steps] ** 2) * 1 / trading_days
        period_sigma = volatility[n_steps] * np.sqrt(1 / trading_days)
        random_return = np.random.normal(period_rate, period_sigma)
        ts = np.append(ts, np.exp(log_price[-1] + random_return))  # Generate an estimated price with random return

        n_steps += 1

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

    rebuild_rate: int
        Rate to rebuild ARMA-GARCH model

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

    def __init__(self, ticker, period='max', model='Returns', horizon=365, trading_days=365, rebuild_rate=1,
                 risk_free_rate=.03, options_info=None, arima=None, arch_garch=None):

        self.trading_days = trading_days
        self.horizon = horizon
        self.rebuild_rate = rebuild_rate
        self.model = model
        self.options_info = options_info
        self.risk_free_rate = risk_free_rate
        self.simulated_series = []
        self.results = []

        if model == 'Options' and options_info is None:
            print('Modeling options requires a dictionary')
            print("dict(type='Asian', strike='geometric', interval=4)")
            print("dict(type='European', strike=54.96, interval=None)")
            raise TypeError

        self.ticker = yf.Ticker(ticker)

        if isinstance(period, dict):
            if 'start' not in period or 'end' not in period:
                print('Period should have start and end dates in %Y-%m-%d format')
                raise TypeError
            self.ts = self.ticker.history(start=period['start'], end=period['end'])
        else:
            self.ts = self.ticker.history(period=period)

        if arch_garch is None:
            self.arma_garch = dict(vol='GARCH', p=1, q=1, rescale=True, dist='normal')
        else:
            self.arma_garch = arch_garch

        if arima is None:
            self.arima = dict(information_criterion='bic')
        else:
            self.arima = arima

    def SimulateOnce(self):
        """ Simulate one price movement for the given horizon period"""

        simulated_series = SimulateGarch(self.ts['Close'], self.horizon, self.trading_days, self.rebuild_rate,
                                         self.risk_free_rate, self.arima, self.arma_garch)
        self.simulated_series.append(simulated_series)

        if self.model == 'Returns':
            result = self.ts['Close'][-1] - simulated_series[-1]

        elif self.model == 'Options':
            result = SimulateOptions(simulated_series, self.options_info['type'], self.options_info['strike'],
                                     self.options_info['interval'])

        # Return result discounted by the risk-free rate
        return result * np.exp(-self.risk_free_rate * (1 / self.trading_days))

    def Simulation_Statistics(self):
        """Generates the relevant plots and statistics for the Monte Carlo simulation results"""

        self.results = np.array(self.results)
        vtime = self.ts.index[-1] - self.ts.index[0]
        years = int(vtime.days / 365)
        months = int((vtime.days % 365) / 30)
        days = int((vtime.days % 365) % 30)

        print(self.elapsed_time)
        print('-' * len(self.elapsed_time))
        print('Simulated prices from {} year(s), {} month(s), and {} day(s) of historical data'.format(
            years, months, days))
        print('Average Profit/Loss: ${:,.2f}'.format(np.mean(self.results)))
        print('Profit/Loss Ranges from ${:,.2f} - ${:,.2f}'.format(np.min(self.results), np.max(self.results)))
        print('Probability of Earning a Return = {:.2f}%'.format(((self.results > 0).sum() / len(self.results)) * 100))
        print('The VaR at 95% Confidence is: ${:,.2f}'.format(self.var()))
        print('-' * len(self.elapsed_time))

        fig, axs = plt.subplots(1, 2, figsize=(13 * 1.10, 7 * 1.10))
        plot_histogram(self.results, axs[0])
        plot_series(self.simulated_series, self.ts.index[-1], axs[1])

        if 'name' not in self.ticker.info:
            name = self.ticker.ticker
        else:
            name = self.ticker.info['name']

        fig.suptitle('{} Monte Carlo\nRan {} Simulation(s) of {} day(s)'.format(
            name, self.sim_count, self.trading_days), fontsize=30, fontweight='bold')

        fig.tight_layout()
        plt.show()
