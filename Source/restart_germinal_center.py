import pickle
import os

from germinal_center import *


if __name__ == "__main__":
    save_data = os.listdir("Restart data/")
    most_recent = save_data[-1]
    parameters_file = open("Restart data/{}".format(most_recent), "rb")

    parameters = pickle.load(parameters_file)

    hyphasma(parameters)