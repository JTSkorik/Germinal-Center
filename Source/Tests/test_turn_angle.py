from Source.germinal_center import *
import numpy as np

def test_turn_angle():
    # Tests if turn_angle turns to NaN.
    for i in range(100):
        parameters = Params()
        output = Out(parameters)
        initialise_cells(parameters, output)
        for key in list(output.polarity.keys()):
            initial_vector = output.polarity[key].copy()
            turn_angle(key, np.random.rand(), output)
            assert not np.isnan(output.polarity[key]).any(), "There is a NaN with {} resulting in {}".format(
                initial_vector, output.polarity[key])
