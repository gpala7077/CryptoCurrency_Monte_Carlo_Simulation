import numpy as np


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
    def RunSimulation(self, simCount=100000):
        self.results = []  # Array to hold the results

        # Now, we set up the simulation loop
        for k in range(simCount):
            x = self.SimulateOnce()  # Run the simulation
            self.results.append(x)  # Add the result to the array

        # Retrun the mean result (we will get more information on this shortly)
        return np.mean(self.results)
