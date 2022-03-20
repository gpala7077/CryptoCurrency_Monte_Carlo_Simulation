import scipy.stats as stats


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
