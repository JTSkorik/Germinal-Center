from germinal_center import Params, initialise_cells, turn_angle
import numpy as np


def test_turn_angle():
    for i in range(10):
        parameters = Params()
        initialise_cells(parameters)
        for key in list(parameters.polarity.keys()):
            initial_vector = parameters.polarity[key].copy()
            turn_angle(key, np.random.rand(), parameters)
            assert not np.isnan(parameters.polarity[key]).any(), "There is a NaN with {} resulting in {}".format(
                initial_vector, parameters.polarity[key])
