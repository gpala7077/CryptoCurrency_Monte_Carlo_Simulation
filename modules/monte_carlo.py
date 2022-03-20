import numpy as np
import time
import datetime
from modules.options import Option
from modules.arma_garch import Arma_Garch_Modeler, forecast_sigma, simulate_garch
from modules.finance import Financial_Timeseries
from modules.plots import plot_montecarlo_simulation_results
import dill
import scipy.stats as stats


def bootstrap(x, confidence=.95, nSamples=100):
    # Make "nSamples" new datasets by re-sampling x with replacement
    # the size of the samples should be the same as x itself
    means = []
    for k in range(nSamples):
        sample = np.random.choice(x, size=len(x), replace=True)
        means.append(np.mean(sample))
    means.sort()
    leftTail = int(((1.0 - confidence) / 2) * nSamples)
    rightTail = (nSamples - 1) - leftTail
    return means[leftTail], np.mean(x), means[rightTail]


class MonteCarlo:
    """
    The SimulateOnce method is declared as abstract (it doesn't have an implementation
    in the base class) that must be extended/overriden to build the simualtion.
    """

    def simulate_once(self):
        raise NotImplementedError

    def var(self, risk=.05):
        if hasattr(self, "results"):  # See if the results have been calculated
            self.results.sort()  # Sort them
            index = int(len(self.results) * risk)  # Count them and multiply by the risk factor
            return self.results[index]  # Return the value at that index
        else:
            print("RunSimulation must be executed before the method 'var'")
            return 0.0

    def run_simulation(self, sim_count=100000):
        start = time.time()
        self.results = []  # Array to hold the results
        self.sim_count = sim_count
        # Now, we set up the simulation loop
        print('Beginning Simulations...')
        for k in range(sim_count):
            x = self.simulate_once()  # Run the simulation
            self.results.append(x)  # Add the result to the array

            if sim_count <= 10:
                print('Completed Simulation # {}'.format(k))
            elif 10 < sim_count <= 100 and k % 10 == 0:
                print('Completed Simulation # {}'.format(k))
            elif 100 < sim_count <= 1000 and k % 100 == 0:
                print('Completed Simulation # {:,}'.format(k))
            elif 1000 < sim_count <= 1000 and k % 1000 == 0:
                print('Completed Simulation # {:,}'.format(k))
            elif 10000 < sim_count <= 100000 and k % 10000 == 0:
                print('Completed Simulation # {:,}'.format(k))
            elif sim_count > 100000 and k % 100000 == 0:
                print('Completed Simulation # {:,}'.format(k))

        print('Completed Simulation # {:,}'.format(k + 1))
        self.calculate_time(start, time.time())
        # Return the mean result (we will get more information on this shortly)
        return bootstrap(self.results)

    def calculate_time(self, start, end):
        time_elapsed = str(datetime.timedelta(seconds=end - start))
        if 'days' in time_elapsed:
            days, hms = time_elapsed.split(',')
            hms = [float(i) for i in hms.split(':')]
            text = '\n{:,} Simulation(s) Completed in {}, {:.0f} hour(s), {:.0f} minute(s), and {:.2f} second(s)'. \
                format(self.sim_count, *[days] + hms)
        else:
            hms = [float(i) for i in time_elapsed.split(':')]
            text = '\n{:,} Simulation(s) Completed in {:.0f} hour(s), {:.0f} minute(s), and {:.2f} second(s)'.format(
                self.sim_count, *hms)
        self.elapsed_time = text


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

    def __init__(self, ticker=None, period='max', model='Returns', horizon=365, trading_days=365,
                 risk_free_rate=.03, options_info=None, arima=None, arch_garch=None, numba=False, dill=None):

        if dill is not None:
            self.load(dill)
        elif ticker is not None:
            self.trading_days = trading_days
            self.horizon = horizon
            self.model = model
            self.options = Option(**options_info) if options_info is not None else options_info
            self.risk_free_rate = risk_free_rate
            self.data = Financial_Timeseries(ticker, period)
            self.arma_garch = Arma_Garch_Modeler(self.data.transform('Close', 'log returns'), arima, arch_garch)
            self.numba = numba
            self.simulated_series = []
            self.results = []

    def simulate_once(self):
        """ Simulate one price movement for the given horizon period"""
        if self.numba:
            model_parameters = self.arma_garch.fitted_model.params
            omega = model_parameters['omega']
            alphas = np.array(model_parameters[[alpha for alpha in model_parameters.keys() if 'alpha' in alpha]])
            betas = np.array(model_parameters[[beta for beta in model_parameters.keys() if 'beta' in beta]])
            gammas = np.array(model_parameters[[gamma for gamma in model_parameters.keys() if 'gamma' in gamma]])
            p = self.arma_garch.arch_garch['p']
            o = self.arma_garch.arch_garch['o']
            q = self.arma_garch.arch_garch['q']

            simulated_series = simulate_garch(self.data.timeseries['Close'].to_numpy(), self.horizon, self.trading_days,
                                              self.risk_free_rate, p=p, o=o, q=q, omega=omega, alphas=alphas,
                                              betas=betas, gammas=gammas)
        elif not self.numba:

            simulated_series = self.arma_garch.simulate_garch(self.data.timeseries['Close'], self.horizon,
                                                              self.trading_days, self.risk_free_rate)

        self.simulated_series.append(simulated_series)

        if self.model == 'Returns':
            result = self.data.timeseries['Close'][-1] - simulated_series[-1]

        elif self.model == 'Options':
            result = self.options.simulate_options(simulated_series)

        # Return result discounted by the risk-free rate, if no risk-free rate, then return result
        return result * np.exp(
            -self.risk_free_rate * (1 / self.trading_days)) if self.risk_free_rate is None else result

    def simulation_statistics(self, risk=.05, plot_simulated=None):
        """
            Generates the relevant plots and statistics for the Monte Carlo simulation results

        :param risk:
            A float with the range between 0 and 1, indicating the value at risk level

        :param plot_simulated:
            An int that indicates which simulated series to use to build an ARMA-GARCH model to compare with the
            observed series

        :return:
            None
        """

        plot_simulated = np.random.randint(0, len(self.simulated_series)) if plot_simulated is None else plot_simulated
        self.results = np.array(self.results)
        simulated_model = self.arma_garch.arma_garch_model(np.diff(np.log(self.simulated_series[plot_simulated])))

        print(self.elapsed_time)
        print('-' * len(self.elapsed_time))
        print(self.data)
        print('Average Profit/Loss: ${:,.2f}'.format(np.mean(self.results)))
        print('Profit/Loss Ranges from ${:,.2f} - ${:,.2f}'.format(np.min(self.results), np.max(self.results)))
        print('95% Confidence Interval Range: from ${:,.2f} - ${:,.2f}'.format(
            *stats.norm.interval(alpha=.95, loc=np.mean(self.results),
                                 scale=stats.sem(self.results))))
        print('Probability of Earning a Return = {:.2f}%'.format(((self.results > 0).sum() / len(self.results)) * 100))
        print('The VaR at 95% Confidence is: ${:,.2f}'.format(self.var(risk)))
        print('-' * len(self.elapsed_time))

        plot_montecarlo_simulation_results(
            f_timeseries=self.data,
            results=self.results,
            simulated_series=self.simulated_series,
            observed_model=self.arma_garch.fitted_model,
            simulated_model=simulated_model
        )

    def save(self, folder=None):
        with open('{}{}_{}_sims_{}_days{}_{}.dill'.format('' if folder is None else '{}/'.format(folder), self.model,
                                                          self.sim_count, self.trading_days,
                                                          '' if not self.numba else '_Numba',
                                                          datetime.datetime.now().strftime('%Y-%m-%d')), 'wb') as f:
            dill.dump(self.__dict__, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.__dict__.update(dill.load(f))
