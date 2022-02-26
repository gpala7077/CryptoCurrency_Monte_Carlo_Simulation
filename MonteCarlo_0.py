import numpy as np
import time
import datetime


# A function that helps us compute the statistics for the MonteCarlo simulation.
# Notice that the default confidence interval is the 68th percentile which is
# equivalent to one standard-deviation if the distribution is sufficiently close
# to normal.
def bootstrap(x, confidence=.95, n_samples=100):
    # Make "nSamples" new datasets by re-sampling x with replacement
    # the size of the samples should be the same as x itself
    means = []
    for k in range(n_samples):
        sample = np.random.choice(x, size=len(x), replace=True)
        means.append(np.mean(sample))
    means.sort()
    left_tail = int(((1.0 - confidence)/2) * n_samples)
    right_tail = (n_samples - 1) - left_tail
    return means[left_tail], np.mean(x), means[right_tail]


# A general MonteCarlo engine that runs a simulation many times and computes the average
# and the error in the average (confidence interval for a certain level). 

class MonteCarlo:
    """
    The SimulateOnce method is declared as abstract (it doesn't have an implementation
    in the base class) that must be extended/overriden to build the simualtion.
    """

    def SimulateOnce(self):
        raise NotImplementedError

    def var(self, risk=.05):
        if hasattr(self, "results"):  # See if the results have been calculated
            self.results.sort()  # Sort them
            index = int(len(self.results) * risk)  # Count them and multiply by the risk factor
            return (self.results[index])  # Return the value at that index
        else:
            print("RunSimulation must be executed before the method 'var'")
            return 0.0

    # For the simplest Monte Carlo simulator, we will simply run the
    # simulation for a specific number of iterrations and then
    # average the results
    def RunSimulation(self, sim_count=100000):
        start = time.time()
        self.results = []  # Array to hold the results
        self.sim_count = sim_count
        # Now, we set up the simulation loop
        print('Beginning Simulations...')
        for k in range(sim_count):
            x = self.SimulateOnce()  # Run the simulation
            self.results.append(x)  # Add the result to the array
            print('Completed Simulation # {}'.format(k + 1))

        self.CalculateTime(start, time.time())
        # Return the mean result (we will get more information on this shortly)
        return np.mean(self.results)

    def CalculateTime(self, start, end):
        time_elapsed = str(datetime.timedelta(seconds=end - start))
        if 'days' in time_elapsed:
            days, hms = time_elapsed.split(',')
            hms = [float(i) for i in hms.split(':')]
            text = '\n{} Simulation(s) Completed in {}, {:.0f} hour(s), {:.0f} minute(s), and {:.2f} second(s)'.format(
                self.sim_count, *[days] + hms)
        else:
            hms = [float(i) for i in time_elapsed.split(':')]
            text = '\n{} Simulation(s) Completed in {:.0f} hour(s), {:.0f} minute(s), and {:.2f} second(s)'.format(
                self.sim_count, *hms)
        self.elapsed_time = text
