# Imports
import pickle
import os
from germinal_center import *

if __name__ == "__main__":
    # Import most recent parameters
    save_data = os.listdir("Restart data/")

    most_recent = save_data[-1]
    parameters_file = open("Restart data/{}".format(most_recent), "rb")
    parameters = pickle.load(parameters_file)

    print(parameters.p_mutation)

    parameters.tmax = 100.0

    # Continue running simulation
    hyphasma(parameters)