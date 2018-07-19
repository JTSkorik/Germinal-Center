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
        self.offset = (self.n / 2 + 0.5, self.n / 2 + 0.5, self.n / 2 + 0.5)  # Amount each co-ordinate moved

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
        self.p_mhc_dep_nhill = 1.0
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
        self.prob_now = self.dt * 9.0
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
    if random.uniform(0, 1) < parameters.p_mutation[cell_id]:
        # Randomly obtain index to mutate
        index = random.choice([0, 1, 2, 3])
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
                # General case, apply either with equal probability.
                if random.uniform(0, 1) < 0.5:
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
    #
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
    cell_type = parameters.type[cell_id]
    cell_state = parameters.state[cell_id]
    cell_pos = parameters.position[cell_id]

    # Update Centrocyte cells based on cxcl13 amounts
    if cell_type == CellType.Centrocyte:
        if cell_state in [CellState.Unselected, CellState.FDCselected]:
            if parameters.grid_cxcl13[cell_pos] > parameters.cxcl13_crit:
                parameters.responsive_to_cxcl13[cell_id] = False
            elif parameters.grid_cxcl13[cell_pos] < parameters.cxcl13_recrit:
                parameters.responsive_to_cxcl13[cell_id] = True

    # Update Centroblast cells based on cxcl12 amounts
    elif cell_type == CellType.Centroblast:
        if parameters.grid_cxcl12[cell_pos] > parameters.cxcl12_crit:
            parameters.responsive_to_cxcl12[cell_id] = False
        elif parameters.grid_cxcl12[cell_pos] < parameters.cxcl12_recrit:
            parameters.responsive_to_cxcl12[cell_id] = True


def move(cell_id, parameters):
    """
    With a given probability, a given cell is moved to an avaiable position in the
    direction of its polarity.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    cell_position = parameters.position[cell_id]
    x, y, z = cell_position

    # Obtain required parameters
    cell_type = parameters.type[cell_id]
    if cell_type == CellType.Centrocyte:
        prob = parameters.plt_centrocyte
        speed = parameters.speed_centrocyte
    elif cell_type == CellType.Centroblast:
        prob = parameters.plt_centroblast
        speed = parameters.speed_centroblast
    elif cell_type == CellType.TCell:
        prob = parameters.plt_tcell
        speed = parameters.speed_tcell
    elif cell_type == CellType.Outcell:
        prob = parameters.plt_outcell
        speed = parameters.speed_outcell
    else:
        prob = None
        speed = None
        print("move: Invalid cell_type, {}".format(cell_type))

    # Calculate new polarity
    if random.uniform(0, 1) < prob:
        # Obtain turning angles
        theta = random.gauss(0, 1)
        phi = random.uniform(0, 2 * math.pi)
        turn_angle(cell_id, theta, phi, parameters)

    # Find CXCL13 influence
    if parameters.responsive_to_cxcl12:
        x_diff = parameters.grid_cxcl12[(x + 1, y, z)] - parameters.grid_cxcl12[(x - 1, y, z)]
        y_diff = parameters.grid_cxcl12[(x, y + 1, z)] - parameters.grid_cxcl12[(x, y - 1, z)]
        z_diff = parameters.grid_cxcl12[(x, y, z + 1)] - parameters.grid_cxcl12[(x, y, z - 1)]

        gradient_cxcl12 = np.array([x_diff, y_diff, z_diff]) / (2 * parameters.dx)
        mag_gradient_cxcl12 = np.linalg.norm(gradient_cxcl12)
        chemo_factor = (parameters.chemo_max / (
            1 + math.exp(parameters.chemo_steep * (parameters.chemo_half - 2 * parameters.dx * mag_gradient_cxcl12))))
        parameters.polarity[cell_id] += chemo_factor * gradient_cxcl12

    # Find CXCL13 influence
    if parameters.responsive_to_cxcl13:
        x_diff = parameters.grid_cxcl13[(x + 1, y, z)] - parameters.grid_cxcl13[(x - 1, y, z)]
        y_diff = parameters.grid_cxcl13[(x, y + 1, z)] - parameters.grid_cxcl13[(x, y - 1, z)]
        z_diff = parameters.grid_cxcl13[(x, y, z + 1)] - parameters.grid_cxcl13[(x, y, z - 1)]

        gradient_cxcl13 = np.array([x_diff, y_diff, z_diff]) / (2 * parameters.dx)
        mag_gradient_cxcl13 = np.linalg.norm(gradient_cxcl13)
        chemo_factor = (parameters.chemo_max / (
            1 + math.exp(parameters.chemo_steep * (parameters.chemo_half - 2 * parameters.dx * mag_gradient_cxcl13))))
        parameters.polarity[cell_id] += chemo_factor * gradient_cxcl13

    # T Cell specific influence
    if cell_type == CellType.TCell:
        parameters.polarity[cell_id] = (1.0 - parameters.north_weight) * parameters.polarity[
            cell_id] + parameters.north_weight * parameters.north

    parameters.polarity[cell_id] = parameters.polarity[cell_id] / np.linalg.norm(parameters.polarity[cell_id])

    # Move cell with probability p_difu
    p_difu = speed * parameters.dt / parameters.dx

    if random.uniform(0, 1) < p_difu:
        # Find possible new positions based on order of best preference
        wanted_position = np.array(cell_position) + parameters.polarity[cell_id]
        neighbours = [np.array(movement) + np.array(cell_position) for movement in parameters.possible_neighbours if
                      parameters.grid_id[tuple(np.array(movement) + np.array(cell_position))] != -1]
        neighbours.sort(key=lambda possible_position: np.linalg.norm(possible_position - wanted_position))

        # Move the cell to best available position that isn't against direction of polarity
        count = 0
        moved = False
        while not moved and count <= 9:
            new_cell_position = tuple(neighbours[count])
            if parameters.grid_id[new_cell_position] is None:
                parameters.position[cell_id] = new_cell_position

                parameters.grid_id[new_cell_position] = cell_id
                parameters.grid_type[new_cell_position] = cell_type
                parameters.grid_id[new_cell_position] = None
                parameters.grid_type[new_cell_position] = None

                moved = True
            count += 1

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
    if parameters.num_divisions_to_do == 0:
        parameters.state[cell_id] = CellState.stop_dividing
    else:
        parameters.state[cell_id] = CellState.cb_G1
        parameters.cycle_start_time[cell_id] = 0
        parameters.end_of_this_phase[cell_id] = get_duration(CellState.cb_G1)


def progress_cycle(cell_id, parameters):
    """
    Given enough time has passed, the given cell (centroblast) will transition to its
    next state.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    parameters.cycle_start_time[cell_id] += parameters.dt
    cell_cycle_start_time = parameters.cycle_start_time[cell_id]
    cell_state = parameters.state[cell_id]

    # Progress cell to its next state
    if cell_cycle_start_time > parameters.end_of_this_phase[cell_id]:
        if cell_state == CellState.cb_G1:
            parameters.state[cell_id] = CellState.cb_S
        elif cell_state == CellState.cb_S:
            parameters.state[cell_id] = CellState.cb_G2
        elif cell_state == CellState.cb_G2:
            parameters.state[cell_id] = CellState.cb_M
        elif cell_state == CellState.cb_M:
            parameters.state[cell_id] = CellState.cb_divide

            # Find time until next end of new state and reset cycle start time
            if parameters.state[cell_id] not in [CellState.cb_divide, CellState.stop_dividing]:
                parameters.end_of_this_phase[cell_id] = get_duration(parameters.state[cell_id])
                parameters.cycle_start_time[cell_id] = 0


def divide_and_mutate(cell_id, parameters):
    """
    With a given probability, a given cell (centroblast) will divide into a surrounding
    position and both cells with attempt to mutate.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """

    if random.uniform(0, 1) < parameters.prob_now:
        old_cell_pos = parameters.position[cell_id]
        # Find empty positions around cell
        empty_neighbours = [tuple(np.array(old_cell_pos) + np.array(possible_neighbour)) for possible_neighbour in
                            parameters.possible_neighbours if
                            np.linalg.norm(
                                np.asarray(possible_neighbour) + np.asarray(old_cell_pos) - np.array(
                                    parameters.offset)) <= (parameters.n / 2) and
                            old_cell_pos[2] + possible_neighbour[2] > parameters.n / 2]

        if empty_neighbours:
            new_cell_pos = random.choice(empty_neighbours)

            # Obtain new ID for the new cell and copy over the properties from the old cell
            new_cell_id = parameters.available_cell_ids.pop()
            parameters.list_cb.append(new_cell_id)

            parameters.type[new_cell_id] = CellType.Centroblast
            parameters.position[new_cell_id] = new_cell_pos
            parameters.state[new_cell_id] = None
            parameters.bcr[new_cell_id] = parameters.bcr[cell_id]
            parameters.polarity[new_cell_id] = np.array(parameters.polarity[cell_id])
            parameters.responsive_to_cxcl12[new_cell_id] = True
            parameters.responsive_to_cxcl13[new_cell_id] = False
            parameters.num_divisions_to_do[new_cell_id] = parameters.num_divisions_to_do[cell_id] - 1
            parameters.p_mutation[new_cell_id] = parameters.p_mutation[new_cell_id]
            parameters.i_am_high_ag[new_cell_id] = False
            parameters.retained_ag[new_cell_id] = None
            parameters.cycle_start_time[new_cell_id] = None
            parameters.end_of_this_phase[new_cell_id] = None

            # Update the cell that was divided
            parameters.num_divisions_to_do[new_cell_id] -= 1
            parameters.i_am_high_ag[new_cell_id] = False

            # Update grid parameters
            parameters.grid_id[new_cell_pos] = new_cell_id
            parameters.grid_type[new_cell_id] = CellType.Centroblast

            # Initiate cycles for each cell
            initiate_cycle(cell_id, parameters)
            initiate_cycle(new_cell_id, parameters)

            # Mutate cells
            if parameters.t > parameters.mutation_start_time:
                mutate(cell_id, parameters)
                mutate(new_cell_id, parameters)

            # Find amount of retained antigen for each cell.
            if random.uniform(0, 1) < parameters.prob_divide_ag_asymmetric:
                if parameters.retained_ag[cell_id] == 0:
                    parameters.retained_ag[new_cell_id] = 0
                else:
                    sep = random.gauss(parameters.polarity_index, 1)
                    while sep < 0 or sep > 1:
                        sep = random.gauss(parameters.polarity_index, 1)

                    parameters.retained_ag[new_cell_id] = sep * parameters.retained_ag[cell_id]
                    parameters.reatined_ag[cell_id] = (1 - sep) * parameters.retained_ag[cell_id]

                    if sep > 0.5:
                        parameters.i_am_high_ag[new_cell_id] = True
                    else:
                        parameters.i_am_high_ag[cell_id] = True

            else:
                parameters.retained_ag[cell_id] = parameters.retained_ag[cell_id] / 2
                parameters.retained_ag[new_cell_id] = parameters.retained_ag[cell_id]


def progress_fdc_selection(cell_id, parameters):
    """
    Allows for a given cell (centrocyte) to collect antigen from neighbouring F cells
    or Fragments.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    if parameters.state[cell_id] == CellState.Unselected:
        # Progress selected clock and check if able to collect antigen.
        parameters.selected_clock[cell_id] += parameters.dt
        if parameters.selected_clock[cell_id] <= parameters.collected_fdc_period:
            parameters.clock[cell_id] += parameters.dt

            if parameters.clock[cell_id] > parameters.test_delay:
                parameters.selectable[cell_id] = True

                # Find neighbouring frag component with largest amount of antigen.
                frag_max = None
                frag_max_id = None
                cell_pos = parameters.position[cell_id]
                for neighbour in parameters.possible_neighbours:
                    neighbour_pos = tuple(np.array(cell_pos) + np.array(neighbour))
                    if parameters.grid_type[neighbour_pos] in [CellType.Fragment, CellType.FCell]:
                        frag_id = parameters.grid_id[neighbour_pos]
                        if parameters.antigen_amount(frag_id) > frag_max:
                            frag_max = parameters.antigen_amount[frag_id]
                            frag_max_id = frag_id

                p_bind = affinity(parameters.bcr[cell_id]) * frag_max / parameters.antigen_saturation

                # Bind cell and fragment with probability p_bind or reset the clock.
                if random.uniform(0, 1) < p_bind:
                    parameters.state[cell_id] = CellState.FDCcontact
                    parameters.frag_contact[cell_id] = frag_max_id
                else:
                    parameters.clock[cell_id] = 0
                    parameters.selectable[cell_id] = False
            else:
                # Cell dies if it doesn't get any contacts.
                if parameters.num_fdc_contacts[cell_id] == 0:
                    parameters.state[id] = CellState.Apoptosis
                else:
                    parameters.State[id] = CellState.FDCselected

        # If has contact, absorb antigen
        elif parameters.state[cell_id] == CellState.Contact:
            parameters.selected_clock[cell_id] += parameters.dt
            if random.uniform(0, 1) < parameters.p_sel:
                parameters.num_fdc_contacts += 1
                parameters.state[cell_id] = CellState.Unselected
                parameters.clock[cell_id] = 0
                parameters.selectable[cell_id] = False

                frag_cell_id = parameters.frag_contact[cell_id]
                parameters.antigen_amount[frag_cell_id] -= 1


def progress_tcell_selection(cell_id, parameters):
    """

    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    cell_state = parameters.CellState[cell_id]

    if cell_state == CellState.FDCselected:
        # Find if there is a neighbouring T cell.
        cell_position = parameters.position[cell_id]
        for neighbour in parameters.possible_neighbours:
            neighbour_pos = tuple(np.array(cell_position) + np.array(neighbour))
            # Record neighbouring contact
            if parameters.grid_type[neighbour_pos] == CellType.TCell and parameters.State[id] != CellState.TCcontact:
                update_tcell(cell_id, parameters.grid_id[neighbour_pos], parameters)
                parameters.state[cell_id] = CellState.TCcontact
                parameters.tc_clock[cell_id] = 0
                parameters.tc_signal_duration[cell_id] = 0

    elif cell_state == CellState.TCcontact:
        parameters.tc_clock[cell_id] += parameters.dt
        tcell_id = parameters.tcell_contact[cell_id]
        # Check is current cell has least amount of antigens compared to T cells neighbouring cells.
        lowest_antigen = True
        for bcell_id in parameters.bcell_contacts[tcell_id]:
            if cell_id != bcell_id and parameters.retained_ag[cell_id] <= parameters.retained_ag[bcell_id]:
                lowest_antigen = False

        if lowest_antigen:
            parameters.tc_signal_duration[cell_id] += parameters.dt

        if parameters.tc_signal_duration[cell_id] > parameters.tc_rescue_time:
            parameters.state[cell_id] = CellState.Selected
            parameters.selected_clock[cell_id] = 0
            rand = random.uniform(0, 1)
            parameters.individual_dif_delay[cell_id] = parameters.dif_delay * (1 + 0.1 * math.log(1 - rand) / rand)
            liberate_tcell(cell_id, parameters.tcell_contact[cell_id], parameters)

    elif cell_state == CellState.Selected:
        parameters.selected_clock[cell_id] += parameters.dt
        if parameters.selected_clock[cell_id] > parameters.individual_dif_delay[cell_id]:
            if random.uniform(0, 1) < parameters.prob_dif_to_out:
                differ_to_out(cell_id, parameters)
            else:
                differ_to_cb(cell_id, parameters)


def update_tcell(bcell_id, tcell_id, parameters):
    """
    Updates the states of given b and t cells to record that they are touching/interacting.
    :param bcell_id: integer, determines which b cell in the population we are considering.
    :param tcell_id: integer, determines which t cell in the population we are considering.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """

    parameters.bcell_contacts[tcell_id].append(bcell_id)
    parameters.tcell_contact[bcell_id] = tcell_id
    parameters.state[tcell_id] = CellState.TC_CC_Contact


def liberate_tcell(bcell_id, tcell_id, parameters):
    """
    Updates the states of given b and t cells to record that they are no longer touching/
    interacting.
    :param bcell_id: integer, determines which b cell in the population we are considering.
    :param tcell_id: interger, determines which t cell in the population we are considering.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    parameters.bcell_contacts[tcell_id].remove(bcell_id)
    if not parameters.bcell_contacts[tcell_id]:
        parameters.state[tcell_id] = CellState.TCnormal
    parameters.tcell_contact[bcell_id] = None


def differ_to_out(cell_id, parameters):
    """
    Transitions a given cell from centroblast or centrocyte into an Outcell.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    parameters.list_outells.append(cell_id)
    parameters.num_bcr_outcells[parameters.bcr[cell_id]] += 1
    # Update cell and grid properties
    parameters.type[cell_id] = CellType.Outcell
    parameters.responsive_to_cxcl12 = None
    parameters.responsive_to_cxcl13 = None
    parameters.grid_type[parameters.position[cell_id]] = CellType.Outcell
    initiate_chemokine_receptors(cell_id, parameters)


def differ_to_cb(cell_id, parameters):
    """
    Transitions a given cell from centrocyte to centroblast.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    parameters.list_cb.append(cell_id)

    # Update cell and grid properties
    parameters.grid_type[parameters.position[cell_id]] = CellType.Centroblast

    parameters.type[cell_id] = CellType.Centroblast
    parameters.responsive_to_cxcl12[cell_id] = None
    parameters.responsive_to_cxcl13[cell_id] = None
    parameters.num_divisions_to_do[cell_id] = None
    parameters.p_mutation[cell_id] = None
    parameters.i_am_high_ag[cell_id] = True
    parameters.retained_ag[cell_id] = parameters.num_fdc_contacts[cell_id]
    parameters.cycle_start_time[cell_id] = None
    parameters.end_of_this_pahase[cell_id] = None

    # Find number of divisions remaining
    ag_factor = parameters.num_fdc_contacts[cell_id] ** parameters.p_mhc_dep_nhill
    parameters.num_divisions_to_do[cell_id] = parameters.p_mhc_dep_min + (
                                                                             parameters.p_mhc_dep_max - parameters.p_mhc_dep_min) * ag_factor / (
                                                                             ag_factor + parameters.p_mhc_depk ** parameters.p_mhc_dep_nhill)

    # Find new probability of mutation
    parameters.p_mutation[cell_id] = p_mut(parameters.t) + parameters.prob_mut_after_selection - p_mut(
        parameters.t) * affinity(parameters.bcr[cell_id]) ** parameters.prob_mut_affinity_exponent

    # Initiate cell
    initiate_chemokine_receptors(cell_id, parameters)
    initiate_cycle(cell_id, parameters)


def differ_to_cc(cell_id, parameters):
    """
    Transitions a given cell from centroblast to centrocyte.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    parameters.list_cc.append(cell_id)
    old_retained_ag = parameters.retained_ag[cell_id]

    # Update cell and grid properties
    parameters.grid_type[parameters.position[cell_id]] = CellType.Centrocyte

    parameters.type[cell_id] = CellType.Centrocyte
    parameters.state[cell_id] = CellState.Unselected
    parameters.responsive_to_cxcl12[cell_id] = None
    parameters.responsive_to_cxcl13[cell_id] = None
    parameters.selected_clock[cell_id] = 0.0
    parameters.clock[cell_id] = 0.0
    parameters.selectable[cell_id] = False
    parameters.frag_contact[cell_id] = None
    parameters.num_fdc_contacts[cell_id] = 0
    parameters.tc_clock[cell_id] = None
    parameters.tc_signal_duration[cell_id] = None
    parameters.individual_dif_delay[cell_id] = None
    parameters.tcell_contact[cell_id] = None

    # Initialise cell and set amount of fdc contacts
    initiate_chemokine_receptors(cell_id, parameters)
    if parameters.delete_ag_in_fresh_cc:
        parameters.num_fdc_contacts[cell_id] = 0
    else:
        parameters.num_fdc_contacts[cell_id] = math.floor(old_retained_ag + 0.5)


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
    if time > 24:
        return 0.5
    else:
        return 0.0


def get_duration(state):
    """
    Uses Guassian random variable to determine how long a cell will stay remain
    in its current state.
    :param state: enumeration, state of the cell being considered.
    :return duration: float, sample from a Guassian random variable.
    """
    sigma = 1
    if state == CellState.cb_G1:
        mu = 2.0
    elif state == CellState.cb_S:
        mu = 1.0
    elif state == CellState.cb_G2:
        mu = 2.5
    elif state == CellState.cb_M:
        mu = 0.5
    else:
        mu = None
        print("getDuration: Invalid cell state, {}".format(state))

    # Small chance duration is negative, ensure we return position value
    duration = random.gauss(mu, sigma)
    while duration < 0:
        duration = random.gauss(mu, sigma)

    return duration


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
