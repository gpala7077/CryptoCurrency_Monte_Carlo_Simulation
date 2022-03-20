from modules.monte_carlo import Timeseries_MonteCarlo

# Set up Monte Carlo Configuration #####################################################################################
trading_days = 365
horizon = 365
model = 'Returns'  # 'Returns' or 'Options'
simulations = 10000
risk_free_rate = .03
ticker = 'BTC-USD'
period = 'max'  # Can also have a start/end data,  dict(start='2000-01-01',end='2015-01-01')
numba = True  # True if you wish to leverage numba, False if not

# Configure ARMA-GARCH
arch_garch = dict(vol='GARCH', p=1, q=1, o=0, mean="Zero", rescale=True,
                  dist='normal')  # These are default values, can be changed
arima = dict(information_criterion='bic')  # These are default values, can be changed

# Configure 'Options' if choosing Options to evaluate results
options_info = dict(options_type='Asian', strike_price='geometric', call=True, contract_price=5,
                    interval=4)  # Only necessary if model='Options', can be European or Asian.
########################################################################################################################


# Monte Carlo Simulation A
ts_1 = Timeseries_MonteCarlo(ticker=ticker, period=period, model=model, horizon=horizon,
                             trading_days=trading_days,
                             options_info=options_info, risk_free_rate=risk_free_rate, arima=arima,
                             arch_garch=arch_garch, numba=numba)
ts_1.run_simulation(simulations)
ts_1.simulation_statistics()

# Monte Carlo Simulation B
numba = False
simulations = 100

ts_2 = Timeseries_MonteCarlo(ticker=ticker, period=period, model=model, horizon=horizon,
                             trading_days=trading_days,
                             options_info=options_info, risk_free_rate=risk_free_rate, arima=arima,
                             arch_garch=arch_garch, numba=numba)
ts_2.run_simulation(simulations)
ts_2.simulation_statistics()
