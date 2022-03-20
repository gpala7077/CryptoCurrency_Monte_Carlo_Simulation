import pmdarima
from arch import arch_model
from numba import jit
import numpy as np


class Arma_Garch_Modeler:
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


@jit(nopython=True)
def forecast_sigma(omega, alphas, betas, gammas, returns, sigmas):
    """
        Forecasts the volatility given the models parameters, previous returns and volatilities

    :param returns:
        A numpy array containing the previous returns

    :param sigmas:
        A numpy array containing the previous volatilities

    :return:
        Returns the new volatility
    """
    returns = np.array(returns)
    sigmas = np.array(sigmas)
    return np.sqrt(omega + np.sum(alphas * (returns ** 2)) + np.sum(betas * (sigmas ** 2)))


@jit(nopython=True)
def simulate_garch(ts, horizon, trading_days, risk_free_rate, p, o, q, omega, alphas, betas, gammas):
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
    sigmas = [log_returns.std()] * q  # Calculate the current volatility

    period_rate = (risk_free_rate - .5 * sigmas[-1] ** 2) * (1 / trading_days)
    period_sigma = sigmas[-1] * np.sqrt(1 / trading_days)
    random_return = np.random.normal(period_rate, period_sigma)
    random_return = [random_return]

    for i in range(horizon):
        period_rate = (risk_free_rate - .5 * sigmas[-1] ** 2) * (1 / trading_days)
        period_sigma = sigmas[-1] * np.sqrt(1 / trading_days)
        random_return.append(np.random.normal(period_rate, period_sigma))

        new_sigma = forecast_sigma(omega=omega, alphas=alphas, betas=betas, gammas=gammas,
                                   returns=random_return[-1],
                                   sigmas=sigmas[-q:])
        sigmas.append(new_sigma)
        # Generate a new price with the random return
        new_price = round(ts[-1] * random_return[-1], 2) if risk_free_rate is None else round(
            np.exp(np.log(ts[-1]) + random_return[-1]), 2)
        ts = np.append(ts, new_price)  # Append new price to the series

    simulated_series = ts[len(actual) - 1:]  # Store the simulated year array

    return simulated_series  # Return simulated series
