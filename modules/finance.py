import yfinance as yf
import numpy as np


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
            print(self.valid_periods)
            raise TypeError

        elif isinstance(period, dict):
            try:
                self.timeseries = self.history(**period)
            except ValueError:
                print('Period should be in %Y-%m-%d format')
                raise ValueError

        elif isinstance(period, str):
            self.timeseries = self.history(period=period)

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
