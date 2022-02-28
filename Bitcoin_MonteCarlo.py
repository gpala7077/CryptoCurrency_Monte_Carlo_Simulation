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


def arma_garch_volatility(series, horizon):
    arima_model_fitted = arima.auto_arima(series, information_criterion='bic')  # Fit an ARIMA model
    arima_residuals = arima_model_fitted.arima_res_.resid  # Retrieve the residuals
    model = arch_model(arima_residuals, vol='GARCH', p=1, q=1, rescale=True)  # Build Garch(1,1) on ARIMA residuals
    fitted_model = model.fit(disp="off")  # Fit the model
    forecast = fitted_model.forecast(horizon=horizon, reindex=False)  # Forecast n-step ahead

    return np.sqrt(forecast.residual_variance.values[0])  # Return volatility forecast


def SimulateGarch(ts, horizon, trading_days, rebuild_rate):
    actual = ts
    n_steps = 0
    volatility = []
    for i in range(horizon):
        log_returns = np.diff(np.log(ts))  # Calculate log returns of the series
        mean_return = log_returns.mean()  # Calculate the mean log return

        if n_steps == len(volatility):
            volatility = arma_garch_volatility(log_returns, rebuild_rate)  # Calculate volatility
            n_steps = 0

        random_return = np.random.normal(  # Generate random return
            (1 + mean_return) ** (1 / trading_days), volatility[n_steps] / np.sqrt(trading_days), 1)

        ts = np.append(ts, ts[-1] * random_return)  # Generate an estimated new price point given the random return

        n_steps += 1

    simulated_series = ts[len(actual) - 1:]  # Store the simulated year array

    return simulated_series, ts[-1] - actual[-1]  # Return simulated series and the profit/loss


def geometricAvg(Lst):
    """Returns the geometric mean of a list."""
    product = 1
    size = len(Lst)
    for num in Lst:
        num *= product
    avg = product ** (1/size)
    return avg
    
def arithmeticAvg(Lst):
    """Returns the arithmetic mean of a list."""
    avg = sum(Lst)/len(Lst)
    return avg    

def SimulateOptions(sigma, prices, num_interval=4, risk_fee_rate=.03, avg_method='geometric'):
    """Returns option payoff discounted by risk-free rate.
    
    Sigma is a list of estimated volitility made by GARCH, prices is a list of prices produced with the GARCH volitilities, 
    num_interval is an interger representing the number of intervals to break the period into for finding to ending price of
    each interval to average, the risk free rate representents the risk free rate in decimal form, and the avg_method is the
    method used to average the prices at the end of each interval, 'geometric' or 'arithmetic'.
    """
    days = len(returns)  # Number of days similated
    
    startingPrice = prices[0]  # Starting price of series
    logS = math.log(startingPrice)  
    for vol in sigma:  # Loop through all predicted sigmas from garch
        step = 1/days  # Size of step

        # Update price based on sigmas predicted by GARCH
        periodRate = (risk_fee_rate - .5 * vol**2) * step
        periodSigma = vol * math.sqrt(step)
        logS += rand.normal(periodRate, periodSigma)
    
    # Obtain prices from original similation to be averaged 
    daysInterval = int(days/num_interval)  # Number of days in each interval to average end price
    priceLst = []  # Hold prices to be averaged
    for i in range(1, num_interval):
        Idx = i * daysInterval  # Index for  bitcoin price to be included in average
        price = prices[Idx]
        priceLst.append(price)
        
    # Get average price based on method choosen
    if avg_method == 'arithmetic':
        avg = arithmeticAvg(priceLst)
    else:
        avg = geometricAvg(priceLst)
        
    # Return the payoff discounted by the risk-free rate
    return max(math.exp(logS) - avg, 0) * math.exp(-risk_fee_rate * days)


class TimeSeries_MonteCarlo(MonteCarlo):
    def __init__(self, ts, model='GARCH', horizon=365, trading_days=365, rebuild_rate=1):
        self.ts = ts
        self.trading_days = trading_days
        self.horizon = horizon
        self.rebuild_rate = rebuild_rate
        self.model = model
        self.simulated_series = []
        self.results = []

    def SimulateOnce(self):
        if self.model == 'GARCH':
            simulated_series, result = SimulateGarch(self.ts['Close'], self.horizon, self.trading_days,
                                                     self.rebuild_rate)
            self.simulated_series.append(simulated_series)

        elif self.model == 'Options':
            result = SimulateOptions(self.sigmas,  self.simulated_series)

        return result

    def Simulation_Statistics(self):
        self.results = np.array(self.results)

        print(self.elapsed_time)
        print('-' * len(self.elapsed_time))
        print('Average Profit/Loss: ${:,.2f}'.format(np.mean(self.results)))
        print('Profit/Loss Ranges from ${:,.2f} - ${:,.2f}'.format(np.min(self.results), np.max(self.results)))
        print('Probability of Earning a Return = {:.2f}%'.format(((self.results > 0).sum() / len(self.results)) * 100))
        print('The VaR at 95% Confidence is: ${:,.2f}'.format(self.var()))
        print('-' * len(self.elapsed_time))

        fig, axs = plt.subplots(1, 2, figsize=(13 * 1.10, 7 * 1.10))
        plot_histogram(self.results, axs[0])
        plot_series(self.simulated_series, self.ts.index[-1], axs[1])

        fig.suptitle('Bitcoin Monte Carlo\nRan {} Simulation(s) of {} day(s)'.format(self.sim_count, self.trading_days),
                     fontsize=30, fontweight='bold')
        fig.tight_layout()
        plt.show()
