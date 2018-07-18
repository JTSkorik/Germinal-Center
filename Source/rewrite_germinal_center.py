# Imports
import random
import math
import numpy as np
import itertools
from enum import Enum

# Enumerations for Cell Type and State comparisons
class CellType(Enum):
    """
    Class used to enumurate the possible Types of cell available in the simulation.
    """
    Stromal = 1
    FCell = 2
    Fragment = 3
    TCell = 4
    Centroblast = 5
    Centrocyte = 6
    Outcell = 7


class CellState(Enum):
    """
    Class used to enumurate the possible States each type of cell can be in during simulation.
    """
    # T cells
    TCnormal = 10
    TC_CC_Contact = 11

    # Centroblasts
    cb_G1 = 12
    cb_G0 = 13
    cb_S = 14
    cb_G2 = 15
    cb_M = 16
    stop_dividing = 17
    cb_divide = 18

    # Centrocytes
    Unselected = 19
    FDCcontact = 20
    FDCselected = 21
    TCcontact = 22
    Selected = 23
    Apoptosis = 24

    # Outcell
    Outcell = 25


class Params():
    """
    Class to store all variables to be passed around to each function.
    """

    def __init__(self):
        # Target Antigen
        self.antigen_value = 1234

        # Distance Variables:
        self.n = 16  # Diameter of sphere/GC
        self.all_points = [(x + self.n // 2 + 1, y + self.n // 2 + 1, z + self.n // 2 + 1) for x in
                           range(-self.n // 2, self.n // 2) for y in
                           range(-self.n // 2, self.n // 2)
                           for z in range(-self.n // 2, self.n // 2) if
                           ((x + 0.5) ** 2 + (y + 0.5) ** 2 + (z + 0.5) ** 2) <= (self.n // 2) ** 2]

        self.dark_zone = [point for point in self.all_points if point[2] > self.n // 2]
        self.light_zone = [point for point in self.all_points if point[2] <= self.n // 2]
        self.offset = (self.n / 2 + 0.5, self.n / 2 + 0.5, self.n / 2 + 0.5)    #Amount each co-ordinate moved

        # Spatial step size (micrometers)
        self.dx = 5

        # Time Variables:
        self.dt = 0.002
        self.t = 0.0
        self.tmin = 0.0
        self.tmax = 30.0

        # Available Cell IDs:
        self.available_cell_ids = list(range(len(self.all_points)))

        # Lists to store ID of each cell in each state (and fragments)
        self.list_stromal = []
        self.list_fdc = []
        self.list_cb = []
        self.list_cc = []
        self.list_tc = []
        self.list_outcells = []

        # Initialisation
        self.initial_num_stromal_cells = 30
        self.initial_num_fdc = 20
        self.initial_num_seeder = 3
        self.initial_num_tc = 25

        self.dendrite_length = 8
        self.initial_antigen_amount_per_fdc = 3000

        # Possible initial BCR values
        self.bcr_values_initial = random.sample(range(1000, 10000), 1000)
        self.bcr_values_all = list(self.bcr_values_initial)
        self.num_bcr_outcells = {bcr: 0 for bcr in self.bcr_values_initial}
        self.num_bcr_outcells_produce = {bcr: 0 for bcr in self.bcr_values_initial}
        self.antibody_per_bcr = {bcr: 0 for bcr in self.bcr_values_initial}

        # Numpy Arrays storing what is at each location. Outside of sphere the points take value -1,
        # initially the points inside the sphere take value None.
        self.grid_id = np.full((self.n + 2, self.n + 2, self.n + 2), -1, dtype=object)
        self.grid_type = np.full((self.n + 2, self.n + 2, self.n + 2), -1, dtype=object)
        for point in self.all_points:
            self.grid_id[point] = None
            self.grid_type[point] = None

        # Dictionaries storing amounts of CXCL12 and CXCL13 at each point:


        self.grid_cxcl12 = np.random.uniform(80e-11, 80e-10, (self.n + 2, self.n + 2, self.n + 2))
        self.grid_cxcl12 = np.random.uniform(0.1e-10, 0.1e-9, (self.n + 2, self.n + 2, self.n + 2))

        # dynamic number of divisions:
        self.num_div_initial_cells = 3
        self.p_mhc_dep_hill = 1.0
        self.p_mhc_dep_min = 1.0
        self.p_mhc_dep_max = 6.0
        self.p_mhc_depk = 6.0

        # Production/ Diffusion Rates:
        self.p_mk_cxcl12 = 4e-7
        self.p_mk_cxcl13 = 1e-8
        self.cxcl13_diff_rate = 1000 * 25 * 0.002

        # Persistent Length time (PLT)
        self.plt_centrocyte = 0.025
        self.plt_centroblast = 0.025
        self.plt_tcell = 0.0283
        self.plt_outcell = 0.0125

        # Dynamic update of chemokine receptors
        self.cxcl12_crit = 60.0e-10
        self.cxcl12_recrit = 40.0e-10
        self.cxcl13_crit = 0.8e-10
        self.cxcl13_recrit = 0.6e-10

        # Chemotaxis
        self.chemo_max = 10
        self.chemo_steep = 1e+10
        self.chemo_half = 2e-11
        self.north_weight = 0.1
        self.north = np.array([0, 0, -1])

        # Speed
        self.speed_centrocyte = 7.5
        self.speed_centroblast = 7.5
        self.speed_tcell = 10.0
        self.speed_outcell = 3.0

        # Divide and Mutate
        self.prob_now = self.dt * 9.0 * 10
        self.mutation_start_time = 2.0
        self.polarity_index = 0.88
        self.prob_divide_ag_asymmetric = 0.72
        self.prob_mut_after_selection = 0.0
        self.prob_mut_affinity_exponent = 1.0

        # Differentiation Rates
        self.start_differentiation = 72.0
        self.prob_dif = self.dt * 0.1
        self.delete_ag_in_fresh_cc = True
        self.dif_delay = 6.0

        self.prob_dif_to_out = 0.0

        # Selection Steps
        self.test_delay = 0.02
        self.collect_fdc_period = 0.7
        self.antigen_saturation = 20
        self.p_sel = self.dt * 0.05
        self.tc_time = 0.6
        self.tc_rescue_time = 0.5

        # Antibody
        self.pm_differentiation_rate = 24.0
        self.antibodies_production = 1e-17
        self.v_blood = 1e-2
        self.n_gc = 1000
        self.ab_prod_factor = self.antibodies_production * self.dt * self.n_gc * self.v_blood * 1e15
        self.antibody_degradation = 30
        self.k_on = 3.6e9

        self.exp_min = 5.5
        self.exp_max = 9.5

        # Movements:
        self.possible_neighbours = list(itertools.product([-1, 0, 1], repeat=3))
        self.possible_neighbours.remove((0, 0, 0))

        # For plots/tests:
        self.num_bcells = []
        self.times = []

        # Cell Properties
        # General
        self.type = {}
        self.position = {}
        self.state = {}
        self.bcr = {}
        self.polarity = {}
        self.responsive_to_cxcl12 = {}
        self.responsive_to_cxcl13 = {}

        # Centroblasts
        self.num_divisions_to_do = {}
        self.p_mutation = {}
        self.i_am_high_ag = {}
        self.retained_ag = {}
        self.cycle_start_time = {}
        self.end_of_this_phase = {}

        # Centrocytes
        self.selected_clock = {}
        self.clock = {}
        self.selectable = {}
        self.frag_contact = {}
        self.num_fdc_contacts = {}
        self.tc_clock = {}
        self.tc_signal_duration = {}
        self.individual_dif_delay = {}
        self.tcell_contact = {}

        # F Cells & Fragments
        self.antigen_amount = {}
        self.ic_amount = {}
        self.fragments = {}
        self.parent = {}

        # T Cells
        self.bcell_contacts = {}


# Main functions

def mutate(cell_id, parameters):
    """
    With a given a probabilty, a given cell's BCR value will be mutated.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    # Determine if mutation occurs
    if random.uniform(0,1) < parameters.p_mutation[cell_id]:
        # Randomly obtain index to mutate
        index = random.choice([0,1,2,3])
        value = int(str(parameters.bcr[cell_id])[3 - index])
        # Apply mutation
        if index == 3 and value == 1:
            # Special case where BCR = 1___. Must increase first digit.
            parameters.bcr[cell_id] += 10 ** index
        else:
            # Special cases where value is 0 (min) or 9 (max). Only one possible operation.
            if value == 0:
                parameters.bcr[cell_id] += 10 ** index
            elif value == 9:
                parameters.bcr[cell_id] -= 10 ** index
            else:
                #General case, apply either with equal probability.
                if random.uniform(0,1) < 0.5:
                    parameters.bcr[cell_id] += 10 ** index
                else:
                    parameters.bcr[cell_id] -= 10 ** index

    # If a new BCR value is obtained, we need to start tracking it.
    new_bcr = parameters.bcr[cell_id]
    if new_bcr not in parameters.bcr_values_all:
        parameters.bcr_values_all.append(new_bcr)
        parameters.num_bcr_outcells[new_bcr] = 0
        parameters.num_bcr_outcells_produce[new_bcr] = 0
        parameters.antibody_per_bcr[new_bcr] = 0


def initiate_chemokine_receptors(cell_id, parameters):
    """
    Sets the variables responsive_to_cxcl12 and responsive_to_cxcl13 for given cell.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    cell_type = parameters.type[cell_id]

    if cell_type == CellType.Centroblast:
        parameters.responsive_to_cxcl12[cell_id] = True
        parameters.responsive_to_cxcl13[cell_id] = False

    elif cell_type == CellType.Centrocyte:
        parameters.responsive_to_cxcl12[cell_id] = False
        parameters.responsive_to_cxcl13[cell_id] = True

    elif cell_type == CellType.Outcell:
        parameters.responsive_to_cxcl12[cell_id] = False
        parameters.responsive_to_cxcl13[cell_id] = False

    else:
        print("initiate_chemokine_receptors: Invalid cell_type, {}".format(cell_type))


def update_chemokines_receptors(cell_id, parameters):
    """
    Sets the variables responsive_to_cxcl12 and responsive_to_cxcl13 for given cell
    base on amounts of cxcl12 and cxcl13 nearby cell.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    pass

def move(cell_id, parameters):
    """
    With a given probability, a given cell is moved to an avaiable position in the
    direction of its polarity.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    pass


def turn_angle(cell_id, theta, phi, parameters):
    """
    Incomplete
    Finds the new polarity for a cell based on its current polarity and given turning
    angles, phi and theta.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param theta: float, turning angle.
    :param phi: float, turning angle.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    pass


def initiate_cycle(cell_id, parameters):
    """
    Sets the state for a centroblast cell depending on the number of divisions left.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    pass


def progress_cycle(cell_id, parameters):
    """
    Given enough time has passed, the given cell (centroblast) will transition to its
    next state.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    pass


def divide_and_mutate(cell_id, parameters):
    """
    With a given probability, a given cell (centroblast) will divide into a surrounding
    position and both cells with attempt to mutate.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    pass


def progress_fdc_selection(cell_id, parameters):
    """
    Allows for a given cell (centrocyte) to collect antigen from neighbouring F cells
    or Fragments.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    pass


def progress_tcell_selection(cell_id, parameters):
    """

    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    pass


def update_tcell(cell_id_b, cell_id_t,parameters):
    """
    Updates the states of given b and t cells to record that they are touching/interacting.
    :param cell_id_b: integer, determines which b cell in the population we are considering.
    :param cell_id_t: interger, determines which t cell in the population we are considering.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    pass


def liberate_tcell(cell_id_b, cell_id_t, parameters):
    """
    Updates the states of given b and t cells to record that they are no longer touching/
    interacting.
    :param cell_id_b: integer, determines which b cell in the population we are considering.
    :param cell_id_t: interger, determines which t cell in the population we are considering.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    pass


def differ_to_out(cell_id, parameters):
    """
    Transitions a given cell from centroblast or centrocyte into an Outcell.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    pass



def differ_to_cb(cell_id, parameters):
    """
    Transitions a given cell from centrocyte to centroblast.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    pass


def differ_to_cc(cell_id, parameters):
    """
    Transitions a given cell from centroblast to centrocyte.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    pass


def initialise_cells(parameters):
    """
    Setup for the simulation. Generates initiate cells for the simulation.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    pass

def hyphasma(parameters):
    """
    Main Driver function for the simulation of a Germinal Center.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """


# The following functions are small functions to assist the main functions.
def affinity(bcr):
    """
    Calculates the affinity between the target antigen and a given antigen value.
    :param bcr: 4-digit integer, BCR value for a cell.
    :return: float, affinity between given BCR and target BCR.
    """
    hamming_dist = sum(el1 != el2 for el1, el2 in zip(str(bcr), str(parameters.antigen_value)))
    return math.exp(-(hamming_dist / 2.8) ** 2)


def signal_secretion():
    """

    :return:
    """
    pass


def diffuse_signal():
    """

    :return:
    """
    pass


def k_off(bcr):
    """
    Calculates the value of k_off for a given BCR value.
    :param bcr: 4-digit integer, BCR value for a cell.
    :return: float, k_off value.
    """
    pass

def p_mut(time):
    """
    Determines the probability a cell will mutate without extra influences.
    :param time: float, current time step of simulation.
    :return: float, probability of mutation.
    """

def get_duration(state):
    """
    Uses Guassian random variable to determine how long a cell will stay remain
    in its current state.
    :param state: enumeration, state of the cell being considered.
    :return: float, sample from a Guassian random variable.
    """
    pass


def is_surface_point(position):
    """
    Determines if a given point is on the surface of the germinal center. Define the
    surface to be any point that has a face neighbour outside of the germinal center.
    :param position: 3-tuple, position within the germinal center.
    :return: boolean, where the position is on the surface of the germinal center.
    """

if __name__ == "__main__":
    parameters = Params()
    hyphasma(parameters)
