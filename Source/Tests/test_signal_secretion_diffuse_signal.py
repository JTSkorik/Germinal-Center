from Source.germinal_center import *
import numpy as np

def test_diffuse_signal():
    # Tests if diffuse_signal increases in concentration
    # Generate Parameters and Output objects
    parameters = Params()
    output = Out(parameters)

    for _ in range(10000):
        # Apply function to starting state many times
        """
        Since this process is independent of all others, we can ignore 
        the other processes.
        """
        current_total_cxcl12 = np.sum(output.grid_cxcl12)
        current_total_cxcl13 = np.sum(output.grid_cxcl13)
        diffuse_signal(parameters, output)

        assert current_total_cxcl12 > np.sum(output.grid_cxcl12), "CXCL12 has increased while diffusing{}"
        assert current_total_cxcl13 > np.sum(output.grid_cxcl13), "CXCL13 has increased while diffusing"


"""
How to test signal secretion? 
Apply function for each signal. Ensure other one doesn't change. Make sure overall change isn't too much
"""