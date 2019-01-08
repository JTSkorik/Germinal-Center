# Imports
import random
import math
import numpy as np
import itertools
from enum import Enum
import matplotlib.pyplot as plt
import pickle
import json
import csv
import os
import sys


# Enumerations for Cell Type and State comparisons
class CellType(Enum):
    """
    Class used to enumurate the possible Types of cells available in the simulation.
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

        # Generate list of all possible points in GC
        self.all_points = []
        for x in range(-self.n // 2, self.n // 2):
            for y in range(-self.n // 2, self.n // 2):
                for z in range(-self.n // 2, self.n // 2):
                    if ((x + 0.5) ** 2 + (y + 0.5) ** 2 + (z + 0.5) ** 2) <= (self.n // 2) ** 2:
                        self.all_points.append((x + self.n // 2 + 1, y + self.n // 2 + 1, z + self.n // 2 + 1))

        self.dark_zone = [point for point in self.all_points if point[2] > self.n // 2]
        self.light_zone = [point for point in self.all_points if point[2] <= self.n // 2]
        self.offset = (self.n / 2 + 0.5, self.n / 2 + 0.5, self.n / 2 + 0.5)  # Amount each co-ordinate moved

        # Spatial step size (micrometers)
        self.dx = 5

        # Time Variables:
        self.dt = 0.002
        self.tmin = 0.0
        self.tmax = 30.0

        # Initialisation
        self.initial_num_stromal_cells = 30
        self.initial_num_fdc = 20
        self.initial_num_seeder = 3
        self.initial_num_tcells = 25

        self.dendrite_length = 8
        self.initial_antigen_amount_per_fdc = 3000

        self.bcr_values_initial = random.sample(range(1000, 10000), 1000)

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
        self.start_differentiation = 10.0
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


class Out():
    """
    Class to store all output of the simulation.
    """

    def __init__(self, parameters):
        self.t = 0.0  # Simulation time

        # Counter used for saved data
        self.save_counter = 0

        # Available Cell IDs:
        self.available_cell_ids = list(range(len(parameters.all_points)))

        # Lists to store ID of each cell in each state (and fragments)
        self.list_stromal = []
        self.list_fdc = []
        self.list_cb = []
        self.list_cc = []
        self.list_tc = []
        self.list_outcells = []

        # Possible initial BCR values
        self.bcr_values_all = list(parameters.bcr_values_initial)
        self.num_bcr_outcells = {bcr: 0 for bcr in parameters.bcr_values_initial}
        self.num_bcr_outcells_produce = {bcr: 0 for bcr in parameters.bcr_values_initial}
        self.antibody_per_bcr = {bcr: 0 for bcr in parameters.bcr_values_initial}

        # Numpy Arrays storing what is at each location. Outside of sphere the points take value -1,
        # initially the points inside the sphere take value None.
        self.grid_id = np.full((parameters.n + 2, parameters.n + 2, parameters.n + 2), -1, dtype=object)
        self.grid_type = np.full((parameters.n + 2, parameters.n + 2, parameters.n + 2), -1, dtype=object)
        for point in parameters.all_points:
            self.grid_id[point] = None
            self.grid_type[point] = None

        # Numpy arrays storing amounts of CXCL12 and CXCL13 at each point:
        self.grid_cxcl12 = np.zeros([parameters.n + 2, parameters.n + 2, parameters.n + 2])
        self.grid_cxcl13 = np.zeros([parameters.n + 2, parameters.n + 2, parameters.n + 2])

        # Lower and upper bounds for each end of the GC.
        lower_cxcl12 = 80e-13
        upper_cxcl12 = 80e-7
        lower_cxcl13 = 0.1e-13
        upper_cxcl13 = 0.1e-6

        spread_cxcl12 = np.linspace(lower_cxcl12, upper_cxcl12, parameters.n)
        spread_cxcl13 = np.linspace(lower_cxcl13, upper_cxcl13, parameters.n)

        for i in range(1, parameters.n + 1):
            self.grid_cxcl12[:, :, i] = spread_cxcl12[i - 1]
            noise_cxcl12 = np.random.normal(0, spread_cxcl12[i - 1] / 10, (parameters.n + 2, parameters.n + 2))
            self.grid_cxcl12[:, :, i] += noise_cxcl12

            self.grid_cxcl13[:, :, i] = spread_cxcl13[i - 1]
            noise_cxcl13 = np.random.normal(0, spread_cxcl13[i - 1] / 10, (parameters.n + 2, parameters.n + 2))
            self.grid_cxcl13[:, :, i] += noise_cxcl13

        # Set values outside of GC to zero
        for (x, y, z), value in np.ndenumerate(self.grid_cxcl12):
            if np.linalg.norm(np.array([x, y, z]) - np.array(parameters.offset)) > (parameters.n / 2):
                self.grid_cxcl12[x, y, z] = 0
                self.grid_cxcl13[x, y, z] = 0

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

def mutate(cell_id, parameters, output):
    """
    With a given a probabilty, a given cell's BCR value will be mutated.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    # Determine if mutation occurs
    if random.uniform(0, 1) < output.p_mutation[cell_id]:
        # Randomly obtain index to mutate
        index = random.choice([0, 1, 2, 3])
        value = int(str(output.bcr[cell_id])[3 - index])
        # Apply mutation
        if index == 3 and value == 1:
            # Special case where BCR = 1___. Must increase first digit.
            output.bcr[cell_id] += 10 ** index
        else:
            # Special cases where value is 0 (min) or 9 (max). Only one possible operation.
            if value == 0:
                output.bcr[cell_id] += 10 ** index
            elif value == 9:
                output.bcr[cell_id] -= 10 ** index
            else:
                # General case, apply either with equal probability.
                if random.uniform(0, 1) < 0.5:
                    output.bcr[cell_id] += 10 ** index
                else:
                    output.bcr[cell_id] -= 10 ** index

    # If a new BCR value is obtained, we need to start tracking it.
    new_bcr = output.bcr[cell_id]
    if new_bcr not in output.bcr_values_all:
        output.bcr_values_all.append(new_bcr)
        output.num_bcr_outcells[new_bcr] = 0
        output.num_bcr_outcells_produce[new_bcr] = 0
        output.antibody_per_bcr[new_bcr] = 0


def initiate_chemokine_receptors(cell_id, parameters, output):
    """
    Sets the variables responsive_to_cxcl12 and responsive_to_cxcl13 for given cell.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    cell_type = output.type[cell_id]

    if cell_type == CellType.Centroblast:
        output.responsive_to_cxcl12[cell_id] = True
        output.responsive_to_cxcl13[cell_id] = False

    elif cell_type == CellType.Centrocyte:
        output.responsive_to_cxcl12[cell_id] = False
        output.responsive_to_cxcl13[cell_id] = True

    elif cell_type == CellType.Outcell:
        output.responsive_to_cxcl12[cell_id] = False
        output.responsive_to_cxcl13[cell_id] = False

    else:
        print("initiate_chemokine_receptors: Invalid cell_type, {}".format(cell_type))


def update_chemokines_receptors(cell_id, parameters, output):
    """
    Sets the variables responsive_to_cxcl12 and responsive_to_cxcl13 for given cell
    base on amounts of cxcl12 and cxcl13 nearby cell.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    cell_type = output.type[cell_id]
    cell_state = output.state[cell_id]
    cell_pos = output.position[cell_id]

    # Update Centrocyte cells based on cxcl13 amounts
    if cell_type == CellType.Centrocyte:
        if cell_state in [CellState.Unselected, CellState.FDCselected]:
            if output.grid_cxcl13[cell_pos] > parameters.cxcl13_crit:
                output.responsive_to_cxcl13[cell_id] = False
            elif output.grid_cxcl13[cell_pos] < parameters.cxcl13_recrit:
                output.responsive_to_cxcl13[cell_id] = True

    # Update Centroblast cells based on cxcl12 amounts
    elif cell_type == CellType.Centroblast:
        if output.grid_cxcl12[cell_pos] > parameters.cxcl12_crit:
            output.responsive_to_cxcl12[cell_id] = False
        elif output.grid_cxcl12[cell_pos] < parameters.cxcl12_recrit:
            output.responsive_to_cxcl12[cell_id] = True


def move(cell_id, parameters, output):
    """
    With a given probability, a given cell is moved to an avaiable position in the
    direction of its polarity.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    cell_position = output.position[cell_id]
    x, y, z = cell_position

    # Obtain required parameters
    cell_type = output.type[cell_id]
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
        phi = random.gauss(0, math.pi / 4)
        turn_angle(cell_id, phi, output)

    # Find CXCL13 influence
    if output.responsive_to_cxcl12:
        x_diff = output.grid_cxcl12[(x + 1, y, z)] - output.grid_cxcl12[(x - 1, y, z)]
        y_diff = output.grid_cxcl12[(x, y + 1, z)] - output.grid_cxcl12[(x, y - 1, z)]
        z_diff = output.grid_cxcl12[(x, y, z + 1)] - output.grid_cxcl12[(x, y, z - 1)]

        gradient_cxcl12 = np.array([x_diff, y_diff, z_diff]) / (2 * parameters.dx)
        mag_gradient_cxcl12 = np.linalg.norm(gradient_cxcl12)
        chemo_factor = (parameters.chemo_max / (
                1 + math.exp(
            parameters.chemo_steep * (parameters.chemo_half - 2 * parameters.dx * mag_gradient_cxcl12))))
        output.polarity[cell_id] += chemo_factor * gradient_cxcl12

    # Find CXCL13 influence
    if output.responsive_to_cxcl13:
        x_diff = output.grid_cxcl13[(x + 1, y, z)] - output.grid_cxcl13[(x - 1, y, z)]
        y_diff = output.grid_cxcl13[(x, y + 1, z)] - output.grid_cxcl13[(x, y - 1, z)]
        z_diff = output.grid_cxcl13[(x, y, z + 1)] - output.grid_cxcl13[(x, y, z - 1)]

        gradient_cxcl13 = np.array([x_diff, y_diff, z_diff]) / (2 * parameters.dx)
        mag_gradient_cxcl13 = np.linalg.norm(gradient_cxcl13)
        chemo_factor = (parameters.chemo_max / (
                1 + math.exp(
            parameters.chemo_steep * (parameters.chemo_half - 2 * parameters.dx * mag_gradient_cxcl13))))
        output.polarity[cell_id] += chemo_factor * gradient_cxcl13

    # T Cell specific influence
    if cell_type == CellType.TCell:
        output.polarity[cell_id] = (1.0 - parameters.north_weight) * output.polarity[
            cell_id] + parameters.north_weight * parameters.north

    output.polarity[cell_id] = output.polarity[cell_id] / np.linalg.norm(output.polarity[cell_id])

    # Move cell with probability p_difu
    p_difu = speed * parameters.dt / parameters.dx

    if random.uniform(0, 1) < p_difu:
        # Find all neighbouring positions
        neighbouring_positions = []
        for movement in parameters.possible_neighbours:
            neighbouring_positions.append(np.array(movement) + np.array(cell_position))

        # Find the position wanted based on cells polarity
        wanted_position = np.array(cell_position) + output.polarity[cell_id]

        # Sort all possible neighbouring positions based on preference compared to wanted position
        neighbouring_positions.sort(key=lambda possible_position: np.linalg.norm(possible_position - wanted_position))

        # Move the cell to best available position that isn't against direction of polarity
        # We ensure it isn't against polarity by restricting movement to the 9 best positions
        count = 0
        moved = False
        while not moved and count <= 9:
            new_cell_position = tuple(neighbouring_positions[count])
            if output.grid_id[new_cell_position] is None:
                output.position[cell_id] = new_cell_position

                output.grid_id[new_cell_position] = cell_id
                output.grid_type[new_cell_position] = cell_type
                output.grid_id[new_cell_position] = None
                output.grid_type[new_cell_position] = None

                moved = True
            count += 1


def turn_angle(cell_id, phi, output):
    """
    Finds the new polarity for a cell based on its current polarity and given turning
    angles, phi and theta.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param phi: float, turning angle.
    :param output: Out object, stores all variables in simulation.
    :return:
    """

    polarity = output.polarity[cell_id]

    # Find random vector, r, on surface of unit sphere
    r = np.random.standard_normal(3)
    r = r / np.linalg.norm(r)

    # Find random vector (v) perpendicular to polarity vector using random vector
    v = r - np.dot(r, polarity) * polarity

    # Rotate polarity about v.
    # Create rotation matrix
    R1 = np.cos(phi) * np.identity(3)
    R2 = np.sin(phi) * np.array([[0, -v[2], v[1]],
                                 [v[2], 0, -v[0]],
                                 [-v[1], v[0], 0]])
    R3 = (1 - np.cos(phi)) * np.outer(v, v)
    R = R1 + R2 + R3

    # Apply rotation matrix
    polarity = np.dot(R, polarity)

    # Update polarity parameters (normalise to ensure unit length)
    output.polarity[cell_id] = polarity / np.linalg.norm(polarity)


def initiate_cycle(cell_id, parameters, output):
    """
    Sets the state for a centroblast cell depending on the number of divisions left.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    if output.num_divisions_to_do[cell_id] == 0:
        output.state[cell_id] = CellState.stop_dividing
    else:
        output.state[cell_id] = CellState.cb_G1
        output.cycle_start_time[cell_id] = 0
        output.end_of_this_phase[cell_id] = get_duration(CellState.cb_G1)


def progress_cycle(cell_id, parameters, output):
    """
    Given enough time has passed, the given cell (centroblast) will transition to its
    next state.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    output.cycle_start_time[cell_id] += parameters.dt
    cell_cycle_start_time = output.cycle_start_time[cell_id]
    cell_state = output.state[cell_id]

    # Progress cell to its next state
    if cell_cycle_start_time > output.end_of_this_phase[cell_id]:
        if cell_state == CellState.cb_G1:
            output.state[cell_id] = CellState.cb_S
        elif cell_state == CellState.cb_S:
            output.state[cell_id] = CellState.cb_G2
        elif cell_state == CellState.cb_G2:
            output.state[cell_id] = CellState.cb_M
        elif cell_state == CellState.cb_M:
            output.state[cell_id] = CellState.cb_divide

        # Find time until next end of new state and reset cycle start time
        if output.state[cell_id] not in [CellState.cb_divide, CellState.stop_dividing]:
            output.end_of_this_phase[cell_id] = get_duration(output.state[cell_id])
            output.cycle_start_time[cell_id] = 0


def divide_and_mutate(cell_id, parameters, output):
    """
    With a given probability, a given cell (centroblast) will divide into a surrounding
    position and both cells with attempt to mutate.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """

    if random.uniform(0, 1) < parameters.prob_now:
        old_cell_pos = output.position[cell_id]

        # Find empty positions around cell
        empty_neighbours = []
        for possible_neighbour in parameters.possible_neighbours:
            neighbour = tuple(np.array(old_cell_pos) + np.array(possible_neighbour))
            if output.grid_id[neighbour] is None:
                empty_neighbours.append(neighbour)

        if empty_neighbours:
            new_cell_pos = random.choice(empty_neighbours)

            # Obtain new ID for the new cell and copy over the properties from the old cell
            new_cell_id = output.available_cell_ids.pop()
            output.list_cb.append(new_cell_id)

            output.type[new_cell_id] = CellType.Centroblast
            output.position[new_cell_id] = new_cell_pos
            output.state[new_cell_id] = None
            output.bcr[new_cell_id] = output.bcr[cell_id]
            output.polarity[new_cell_id] = np.array(output.polarity[cell_id])
            output.responsive_to_cxcl12[new_cell_id] = True
            output.responsive_to_cxcl13[new_cell_id] = False
            output.num_divisions_to_do[new_cell_id] = output.num_divisions_to_do[cell_id] - 1
            output.p_mutation[new_cell_id] = output.p_mutation[cell_id]
            output.i_am_high_ag[new_cell_id] = False
            output.retained_ag[new_cell_id] = None
            output.cycle_start_time[new_cell_id] = None
            output.end_of_this_phase[new_cell_id] = None

            # Update the cell that was divided
            output.num_divisions_to_do[new_cell_id] -= 1
            output.i_am_high_ag[new_cell_id] = False

            # Update grid parameters
            output.grid_id[new_cell_pos] = new_cell_id
            output.grid_type[new_cell_pos] = CellType.Centroblast

            # Initiate cycles for each cell
            initiate_cycle(cell_id, parameters, output)
            initiate_cycle(new_cell_id, parameters, output)

            # Mutate cells
            if output.t > parameters.mutation_start_time:
                mutate(cell_id, parameters, output)
                mutate(new_cell_id, parameters, output)

            # Find amount of retained antigen for each cell.
            if random.uniform(0, 1) < parameters.prob_divide_ag_asymmetric:
                if output.retained_ag[cell_id] == 0:
                    output.retained_ag[new_cell_id] = 0
                else:
                    sep = random.gauss(output.polarity_index, 1)
                    while sep < 0 or sep > 1:
                        sep = random.gauss(output.polarity_index, 1)

                    output.retained_ag[new_cell_id] = sep * output.retained_ag[cell_id]
                    output.retained_ag[cell_id] = (1 - sep) * output.retained_ag[cell_id]

                    if sep > 0.5:
                        output.i_am_high_ag[new_cell_id] = True
                    else:
                        output.i_am_high_ag[cell_id] = True

            else:
                output.retained_ag[cell_id] = output.retained_ag[cell_id] / 2
                output.retained_ag[new_cell_id] = output.retained_ag[cell_id]


def progress_fdc_selection(cell_id, parameters, output):
    """
    Allows for a given cell (centrocyte) to collect antigen from neighbouring F cells
    or Fragments.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    if output.state[cell_id] == CellState.Unselected:
        # Progress selected clock and check if able to collect antigen.
        output.selected_clock[cell_id] += parameters.dt
        if output.selected_clock[cell_id] <= parameters.collected_fdc_period:
            output.clock[cell_id] += parameters.dt

            if output.clock[cell_id] > parameters.test_delay:
                output.selectable[cell_id] = True

                # Find neighbouring frag component with largest amount of antigen.
                frag_max = None
                frag_max_id = None
                cell_pos = output.position[cell_id]
                for neighbour in parameters.possible_neighbours:
                    neighbour_pos = tuple(np.array(cell_pos) + np.array(neighbour))
                    if output.grid_type[neighbour_pos] in [CellType.Fragment, CellType.FCell]:
                        frag_id = output.grid_id[neighbour_pos]
                        if output.antigen_amount[frag_id] > frag_max:
                            frag_max = output.antigen_amount[frag_id]
                            frag_max_id = frag_id

                p_bind = affinity(output.bcr[cell_id]) * frag_max / parameters.antigen_saturation

                # Bind cell and fragment with probability p_bind or reset the clock.
                if random.uniform(0, 1) < p_bind:
                    output.state[cell_id] = CellState.FDCcontact
                    output.frag_contact[cell_id] = frag_max_id
                else:
                    output.clock[cell_id] = 0
                    output.selectable[cell_id] = False
            else:
                # Cell dies if it doesn't get any contacts.
                if output.num_fdc_contacts[cell_id] == 0:
                    output.state[id] = CellState.Apoptosis
                else:
                    parameters.State[id] = CellState.FDCselected

        # If has contact, absorb antigen
        elif output.state[cell_id] == CellState.Contact:
            output.selected_clock[cell_id] += parameters.dt
            if random.uniform(0, 1) < parameters.p_sel:
                output.num_fdc_contacts += 1
                output.state[cell_id] = CellState.Unselected
                output.clock[cell_id] = 0
                output.selectable[cell_id] = False

                frag_cell_id = output.frag_contact[cell_id]
                output.antigen_amount[frag_cell_id] -= 1


def progress_tcell_selection(cell_id, parameters, output):
    """
    Progresses the current state of T cells.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    cell_state = output.state[cell_id]

    if cell_state == CellState.FDCselected:
        # Find if there is a neighbouring T cell.
        cell_position = output.position[cell_id]
        for neighbour in parameters.possible_neighbours:
            neighbour_pos = tuple(np.array(cell_position) + np.array(neighbour))
            # Record neighbouring contact
            if output.grid_type[neighbour_pos] == CellType.TCell and output.state[id] != CellState.TCcontact:
                update_tcell(cell_id, output.grid_id[neighbour_pos], parameters, output)
                output.state[cell_id] = CellState.TCcontact
                output.tc_clock[cell_id] = 0
                output.tc_signal_duration[cell_id] = 0

    elif cell_state == CellState.TCcontact:
        output.tc_clock[cell_id] += parameters.dt
        tcell_id = output.tcell_contact[cell_id]
        # Check is current cell has least amount of antigen compared to T cells neighbouring cells.
        lowest_antigen = True
        for bcell_id in output.bcell_contacts[tcell_id]:
            if cell_id != bcell_id and output.retained_ag[cell_id] <= output.retained_ag[bcell_id]:
                lowest_antigen = False

        if lowest_antigen:
            output.tc_signal_duration[cell_id] += parameters.dt

        if output.tc_signal_duration[cell_id] > parameters.tc_rescue_time:
            output.state[cell_id] = CellState.Selected
            output.selected_clock[cell_id] = 0
            rand = random.uniform(0, 1)
            output.individual_dif_delay[cell_id] = parameters.dif_delay * (1 + 0.1 * math.log(1 - rand) / rand)
            liberate_tcell(cell_id, output.tcell_contact[cell_id], parameters, output)

    elif cell_state == CellState.Selected:
        output.selected_clock[cell_id] += parameters.dt
        if output.selected_clock[cell_id] > output.individual_dif_delay[cell_id]:
            if random.uniform(0, 1) < parameters.prob_dif_to_out:
                differ_to_out(cell_id, parameters, output)
            else:
                differ_to_cb(cell_id, parameters, output)


def update_tcell(bcell_id, tcell_id, parameters, output):
    """
    Updates the states of given b and t cells to record that they are touching/interacting.
    :param bcell_id: integer, determines which b cell in the population we are considering.
    :param tcell_id: integer, determines which t cell in the population we are considering.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """

    output.bcell_contacts[tcell_id].append(bcell_id)
    output.tcell_contact[bcell_id] = tcell_id
    output.state[tcell_id] = CellState.TC_CC_Contact


def liberate_tcell(bcell_id, tcell_id, parameters, output):
    """
    Updates the states of given b and t cells to record that they are no longer touching/
    interacting.
    :param bcell_id: integer, determines which b cell in the population we are considering.
    :param tcell_id: integer, determines which t cell in the population we are considering.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    output.bcell_contacts[tcell_id].remove(bcell_id)
    if not output.bcell_contacts[tcell_id]:
        output.state[tcell_id] = CellState.TCnormal
    output.tcell_contact[bcell_id] = None


def differ_to_out(cell_id, parameters, output):
    """
    Transitions a given cell from centroblast or centrocyte into an Outcell.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    output.list_outcells.append(cell_id)
    output.num_bcr_outcells[output.bcr[cell_id]] += 1

    # Update cell and grid properties
    output.type[cell_id] = CellType.Outcell
    output.responsive_to_cxcl12 = None
    output.responsive_to_cxcl13 = None
    output.grid_type[output.position[cell_id]] = CellType.Outcell
    initiate_chemokine_receptors(cell_id, parameters, output)


def differ_to_cb(cell_id, parameters, output):
    """
    Transitions a given cell from centrocyte to centroblast.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    output.list_cb.append(cell_id)

    # Update cell and grid properties
    output.grid_type[output.position[cell_id]] = CellType.Centroblast

    output.type[cell_id] = CellType.Centroblast
    output.responsive_to_cxcl12[cell_id] = None
    output.responsive_to_cxcl13[cell_id] = None
    output.num_divisions_to_do[cell_id] = None
    output.p_mutation[cell_id] = None
    output.i_am_high_ag[cell_id] = True
    output.retained_ag[cell_id] = output.num_fdc_contacts[cell_id]
    output.cycle_start_time[cell_id] = None
    output.end_of_this_phase[cell_id] = None

    # Find number of divisions remaining
    ag_factor = output.num_fdc_contacts[cell_id] ** parameters.p_mhc_dep_nhill
    # Taking floor to ensure its an integer amount
    output.num_divisions_to_do[cell_id] = math.floor(parameters.p_mhc_dep_min + (
            parameters.p_mhc_dep_max - parameters.p_mhc_dep_min) * ag_factor / (
                                                             ag_factor + parameters.p_mhc_depk ** parameters.p_mhc_dep_nhill))

    # Find new probability of mutation
    output.p_mutation[cell_id] = p_mut(output.t) + parameters.prob_mut_after_selection - p_mut(
        output.t) * affinity(output.bcr[cell_id]) ** parameters.prob_mut_affinity_exponent

    # Initiate cell
    initiate_chemokine_receptors(cell_id, parameters, output)
    initiate_cycle(cell_id, parameters, output)


def differ_to_cc(cell_id, parameters, output):
    """
    Transitions a given cell from centroblast to centrocyte.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    output.list_cc.append(cell_id)
    old_retained_ag = output.retained_ag[cell_id]

    # Update cell and grid properties
    output.grid_type[output.position[cell_id]] = CellType.Centrocyte

    output.type[cell_id] = CellType.Centrocyte
    output.state[cell_id] = CellState.Unselected
    output.responsive_to_cxcl12[cell_id] = None
    output.responsive_to_cxcl13[cell_id] = None
    output.selected_clock[cell_id] = 0.0
    output.clock[cell_id] = 0.0
    output.selectable[cell_id] = False
    output.frag_contact[cell_id] = None
    output.num_fdc_contacts[cell_id] = 0
    output.tc_clock[cell_id] = None
    output.tc_signal_duration[cell_id] = None
    output.individual_dif_delay[cell_id] = None
    output.tcell_contact[cell_id] = None

    # Initialise cell and set amount of fdc contacts
    initiate_chemokine_receptors(cell_id, parameters, output)
    if parameters.delete_ag_in_fresh_cc:
        output.num_fdc_contacts[cell_id] = 0
    else:
        output.num_fdc_contacts[cell_id] = math.floor(old_retained_ag + 0.5)


def initialise_cells(parameters, output):
    """
    Setup for the simulation. Generates initiate cells for the simulation.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    # Initialise Stromal Cells
    for _ in range(parameters.initial_num_stromal_cells):
        # Find empty location in dark zone
        cell_position = random.choice(parameters.dark_zone)
        while output.grid_id[cell_position] is not None:
            cell_position = random.choice(parameters.dark_zone)

        cell_id = output.available_cell_ids.pop()

        # Add to appropriate lists and dictionaries
        output.list_stromal.append(cell_id)
        output.type[cell_id] = CellType.Stromal
        output.position[cell_id] = cell_position

        output.grid_id[cell_position] = cell_id
        output.grid_type[cell_position] = CellType.Stromal

    # Initialise F Cells and Fragments
    for _ in range(parameters.initial_num_fdc):
        # Find empty location in light zone
        cell_position = random.choice(parameters.light_zone)
        while output.grid_id[cell_position] is not None:
            cell_position = random.choice(parameters.dark_zone)

        cell_id = output.available_cell_ids.pop()

        # Add to appropriate lists and dictionaries
        output.list_fdc.append(cell_id)

        output.type[cell_id] = CellType.FCell
        output.position[cell_id] = cell_position
        output.antigen_amount[cell_id] = None
        output.ic_amount[cell_id] = 0
        output.fragments[cell_id] = []

        output.grid_id[cell_position] = cell_id
        output.grid_type[cell_position] = CellType.FCell

        # Find fragments for F cell
        # We allow fragments to be in dark zone as it isn't a hard boundary
        fcell_id = cell_id
        fragments = output.fragments[fcell_id]
        x, y, z = cell_position
        for i in range(1, parameters.dendrite_length + 1):
            for fragment_position in [(x + i, y, z), (x - i, y, z), (x, y + i, z), (x, y - i, z), (x, y, z - i),
                                      (x, y, z + i)]:
                # Do nothing if position is outside of GC
                try:
                    if output.grid_id[fragment_position] is None:
                        fragment_id = output.available_cell_ids.pop()
                        fragments.append(fragment_id)

                        output.type[fragment_id] = CellType.Fragment
                        output.position[fragment_id] = fragment_position
                        output.antigen_amount[fragment_id] = None
                        output.ic_amount[fragment_id] = 0
                        output.parent[fragment_id] = fcell_id

                        output.grid_id[fragment_position] = fragment_id
                        output.grid_type[fragment_position] = CellType.Fragment
                except IndexError:
                    pass

            # Assign each fragment an amount of antigen
            fcell_volume = len(fragments) + 1  # +1 accounts for centre
            ag_per_frag = parameters.initial_antigen_amount_per_fdc / fcell_volume
            for cell_id in [fcell_id] + fragments:
                output.antigen_amount[cell_id] = ag_per_frag

    # Initialise Centroblasts
    for _ in range(parameters.initial_num_seeder):
        # Find empty location in light zone
        cell_position = random.choice(parameters.light_zone)
        while output.grid_id[cell_position] is not None:
            cell_position = random.choice(parameters.dark_zone)

        cell_id = output.available_cell_ids.pop()

        # Add to appropriate lists and dictionaries
        output.list_cb.append(cell_id)

        polarity_vector = np.random.standard_normal(3)
        polarity_vector = polarity_vector / np.linalg.norm(polarity_vector)

        output.type[cell_id] = CellType.Centroblast
        output.position[cell_id] = cell_position
        output.state[cell_id] = None
        output.bcr[cell_id] = random.choice(parameters.bcr_values_initial)
        output.polarity[cell_id] = polarity_vector
        output.responsive_to_cxcl12[cell_id] = None
        output.responsive_to_cxcl13[cell_id] = None
        output.num_divisions_to_do[cell_id] = parameters.num_div_initial_cells
        output.p_mutation[cell_id] = p_mut(output.t)
        output.i_am_high_ag[cell_id] = False
        output.retained_ag[cell_id] = 0.0
        output.cycle_start_time[cell_id] = None
        output.end_of_this_phase[cell_id] = None

        # Initialise cells
        initiate_cycle(cell_id, parameters, output)
        initiate_chemokine_receptors(cell_id, parameters, output)

    # Initialise T cells
    for _ in range(parameters.initial_num_tcells):
        # Find empty location in light zone
        cell_position = random.choice(parameters.light_zone)
        while output.grid_id[cell_position] is not None:
            cell_position = random.choice(parameters.dark_zone)

        cell_id = output.available_cell_ids.pop()

        # Add to appropriate lists and dictionaries
        output.list_tc.append(cell_id)

        polarity_vector = np.random.standard_normal(3)
        polarity_vector = polarity_vector / np.linalg.norm(polarity_vector)

        output.type[cell_id] = CellType.TCell
        output.position[cell_id] = cell_position
        output.state[cell_id] = CellState.TCnormal
        output.polarity[cell_id] = polarity_vector
        output.responsive_to_cxcl12[cell_id] = False
        output.responsive_to_cxcl13[cell_id] = False
        output.bcell_contacts[cell_id] = []

        output.grid_id[cell_position] = cell_id
        output.grid_type[cell_position] = CellType.TCell


def hyphasma(parameters, output, filename_output):
    """
    Main Driver function for the simulation of a Germinal Center.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    if output.save_counter == 0:
        initialise_cells(parameters, output)

    while output.t <= parameters.tmax:

        # If 1 simulated hour has elapsed, save current state.
        if output.t >= 1 * output.save_counter:
            # print("Saving current state")
            output.save_counter += 1
            pickle_current_state(output, simulation_name)
            update_out_csv(output, filename_output)

        # print(output.t)
        # Track the number of B cells at each time step. (Used for Testing)
        output.num_bcells.append(len(output.list_cc) + len(output.list_cb))
        # if output.num_bcells[-1] > 3:
        #    print("Number B Cells: {}".format(output.num_bcells[-1]))
        #    print("Number Centroblasts: {}".format(len(output.list_cb)))
        #    print("Number Centrocytes: {}".format(len(output.list_cc)))
        #    print("Number Outcells: {}".format(len(output.list_outcells)))
        output.times.append(output.t)

        # Secrete CXCL12 from Stromal cells
        for cell_id in output.list_stromal:
            signal_secretion(cell_id, parameters)

        random.shuffle(output.list_fdc)
        for cell_id in output.list_fdc:
            # Secrete CXCL13 from F Cells
            signal_secretion(cell_id, parameters)

            # Update antigen amounts for each fragment
            fragments = output.fragments[cell_id]
            for fragment_id in fragments:
                for bcr_seq in output.bcr_values_all:
                    d_ic = parameters.dt * (
                            parameters.k_on * output.antigen_amount[fragment_id] * output.antibody_per_bcr[
                        bcr_seq] - k_off(bcr_seq, parameters) * output.ic_amount[fragment_id])
                    output.antigen_amount[fragment_id] -= d_ic
                    output.ic_amount[fragment_id] += d_ic

        # Diffuse CXCL12/13
        diffuse_signal(parameters)

        # Update the number of outcells and amount of antibody for each CR value.
        for bcr_seq in output.bcr_values_all:
            transfer_t = math.floor(
                output.num_bcr_outcells[bcr_seq] * parameters.pm_differentiation_rate * parameters.dt)
            output.num_bcr_outcells[bcr_seq] -= transfer_t
            output.num_bcr_outcells_produce[bcr_seq] += transfer_t
            output.antibody_per_bcr[bcr_seq] = output.num_bcr_outcells_produce[
                                                   bcr_seq] * parameters.ab_prod_factor - parameters.antibody_degradation * \
                                               output.antibody_per_bcr[
                                                   bcr_seq]

        # Randomly iterate of outcells and move, remove if on surface of GC
        random.shuffle(output.list_outcells)
        outcells_to_remove = []
        for i, cell_id in enumerate(output.list_outcells):
            move(cell_id, parameters, output)
            cell_position = output.position[cell_id]
            if is_surface_point(cell_position, output.grid_id):
                outcells_to_remove.append(i)
                output.available_cell_ids.append(cell_id)
        for i in sorted(outcells_to_remove, reverse=True):
            del (output.list_outcells[i])

        # Randomly iterate over Centroblast cells
        random.shuffle(output.list_cb)
        centroblasts_to_remove = []
        for i, cell_id in enumerate(output.list_cb):
            # Update cell properties
            update_chemokines_receptors(cell_id, parameters, output)
            progress_cycle(cell_id, parameters, output)

            # Attempt to divide if ready
            if output.state[cell_id] == CellState.cb_divide:
                divide_and_mutate(cell_id, parameters, output)

            if output.state[cell_id] == CellState.stop_dividing:
                if random.uniform(0, 1) < parameters.prob_dif:
                    if output.i_am_high_ag[cell_id]:
                        differ_to_out(cell_id, parameters, output)
                        centroblasts_to_remove.append(i)
                    else:
                        differ_to_cc(cell_id, parameters, output)
                        centroblasts_to_remove.append(i)

            # Move allowed cells
            if output.state[cell_id] != CellState.cb_M:
                move(cell_id, parameters, output)
        for i in sorted(centroblasts_to_remove, reverse=True):
            del (output.list_cb[i])

        # Randomly iterated over Centrocyte cells.
        random.shuffle(output.list_cc)
        centrocytes_to_remove = []
        for i, cell_id in enumerate(output.list_cc):
            # Update cell progress
            update_chemokines_receptors(cell_id, parameters, output)
            progress_fdc_selection(cell_id, parameters, output)
            progress_tcell_selection(cell_id, parameters, output)

            # Store index for dead cells for removal
            if output.state[cell_id] == CellState.Apoptosis:
                centrocytes_to_remove.append(i)
                output.available_cell_ids.append(cell_id)
            # Else move possible cells
            elif output.state[cell_id] not in [CellState.FDCcontact, CellState.TCcontact, CellState.Apoptosis]:
                move(cell_id, parameters, output)
        # Remove dead Centrocytes
        for i in sorted(centrocytes_to_remove, reverse=True):
            del (output.list_cc[i])

        # Randomly iterate over T cells and move if not attached to another cell
        random.shuffle(output.list_tc)
        for cell_id in output.list_tc:
            if output.state[cell_id] == CellState.TCnormal:
                move(cell_id, parameters, output)

        output.t += parameters.dt


# Helper functions
def affinity(bcr):
    """
    Calculates the affinity between the target antigen and a given antigen value.
    :param bcr: 4-digit integer, BCR value for a cell.
    :return: float, affinity between given BCR and target BCR.
    """
    hamming_dist = sum(abs(int(el1)-int(el2)) for el1, el2 in zip(str(bcr), str(parameters.antigen_value)))
    return math.exp(-(hamming_dist / 2.8) ** 2)


def signal_secretion(cell_id, parameters):
    """
    Secrets predetermined amound of CXCL1212 from Stromal cells and CXCL13 from Fcells.
    :param cell_id: integer, determines which cell in population we are manipulating.
    :param parameters: params object, stores all parameters and variables in simulation.
    :return:
    """
    cell_position = output.position[cell_id]
    cell_type = output.type[cell_id]

    if cell_type == CellType.Stromal:
        output.grid_cxcl12[tuple(cell_position)] += parameters.p_mk_cxcl12
    elif cell_type == CellType.FCell:
        output.grid_cxcl13[tuple(cell_position)] += parameters.p_mk_cxcl13


def diffuse_signal(parameters):
    """
    How do we do this?
    We're going to make the CXCL12/13 concentrations follow an underlying
    linear relationship with noise.
    Can remove the linear relationship and only look at the noise.
    Now need to find a way to disperse this noise then add linear relationship back.

    Problem 1: We have diffusion constants.
    Look at current time. Find the change of each square based on current state.
    Somehow add randomness. Find net changes in each position and apply it all at once.

    Problem 2: There is nothing that removes CXCL12/13 from the GC.
    Should email Kim about this but will just allow it to be removed along the boundary.

    Problem 3: Boundary values not given for CXCL12.
    Assume zero.

    """

    diff_cxcl12 = np.zeros([parameters.n + 2, parameters.n + 2, parameters.n + 2])
    diff_cxcl13 = np.zeros([parameters.n + 2, parameters.n + 2, parameters.n + 2])

    # Iterate over all points except boundary of matrix which are not part of GC.
    for i in range(1, parameters.n + 1):
        for j in range(1, parameters.n + 1):
            for k in range(1, parameters.n + 1):
                amount_cxcl12 = output.grid_cxcl12[i, j, k]
                amount_cxcl13 = output.grid_cxcl13[i, j, k]

                diff_cxcl12[i, j, k] -= amount_cxcl12 * 2 / 5
                diff_cxcl13[i, j, k] -= amount_cxcl13 * 2 / 5

                # Movements towards dark side
                neighbours = [(i + ii - 1, j + jj - 1, k + 1) for ii in range(3) for jj in range(3)]
                for neighbour in neighbours:
                    diff_cxcl12[neighbour] += amount_cxcl12 / 18
                    diff_cxcl13[neighbour] += amount_cxcl13 / 54

                # Movements towards light side
                neighbours = [(i + ii - 1, j + jj - 1, k - 1) for ii in range(3) for jj in range(3)]
                for neighbour in neighbours:
                    diff_cxcl12[neighbour] += amount_cxcl12 / 54
                    diff_cxcl13[neighbour] += amount_cxcl13 / 18

                # Without Z changing
                neighbours = [(i + ii - 1, j + jj - 1, k) for ii in range(3) for jj in range(3)]
                neighbours.remove((i, j, k))
                for neighbour in neighbours:
                    diff_cxcl12[neighbour] += amount_cxcl12 / 24
                    diff_cxcl13[neighbour] += amount_cxcl13 / 24

    # Make changes:
    output.grid_cxcl12 += diff_cxcl12
    output.grid_cxcl13 += diff_cxcl13

    # Set values outside of GC to zero
    for (x, y, z), value in np.ndenumerate(output.grid_cxcl12):
        if np.linalg.norm(np.array([x, y, z]) - np.array(parameters.offset)) > (parameters.n / 2):
            output.grid_cxcl12[x, y, z] = 0
            output.grid_cxcl13[x, y, z] = 0


def k_off(bcr, parameters):
    """
    Calculates the value of k_off for a given BCR value.
    :param bcr: 4-digit integer, BCR value for a cell.
    :return: float, k_off value.
    """
    hamming_dist = sum(el1 != el2 for el1, el2 in zip(str(bcr), str(parameters.antigen_value)))
    return parameters.k_on / (
            10 ** (parameters.exp_min + math.exp(-(hamming_dist / 2.8) ** 2) * parameters.exp_max - parameters.exp_min))


def p_mut(time):
    """
    Determines the probability a cell will mutate without extra influences.
    :param time: float, current time step of simulation.
    :return: float, probability of mutation.
    """
    if time > 1:
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


def is_surface_point(position, grid_id):
    """
    Determines if a given point is on the surface of the germinal center. Define the
    surface to be any point that has a face neighbour outside of the germinal center.
    :param position: 3-tuple, position within the germinal center.
    :param grid_id: 3d numpy array, position within cells returns what is located at that position.
    :return: boolean, where the position is on the surface of the germinal center.
    """
    position_numpy = np.array(position)
    # Test the main neighbouring points to determine if they are inside the Germinal Center
    for movement in [np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, -1, 0]),
                     np.array([0, 0, 1]), np.array([0, 0, -1])]:
        neighbour_position = position_numpy + movement
        if grid_id[tuple(neighbour_position)] == -1:
            return True

    return False


def params_to_dict(params_instance):
    """
    Converts Params object into a dictionary where the name of each
    variable is the key for its value.
    :param params_instance: Params object
    :return ans: dictionary, all values of parameters as a dictionary with
    """
    ans = dict()
    ans["ab_prod_factor"] = params_instance.ab_prod_factor
    ans["all_points"] = params_instance.all_points
    ans["antibodies_production"] = params_instance.antibodies_production
    ans["antibody_degradation"] = params_instance.antibody_degradation
    ans["antigen_saturation"] = params_instance.antigen_saturation
    ans["antigen_value"] = params_instance.antigen_value
    ans["bcr_values_initial"] = params_instance.bcr_values_initial
    ans["chemo_half"] = params_instance.chemo_half
    ans["chemo_max"] = params_instance.chemo_max
    ans["chemo_steep"] = params_instance.chemo_steep
    ans["collect_fdc_period"] = params_instance.collect_fdc_period
    ans["cxcl12_crit"] = params_instance.cxcl12_crit
    ans["cxcl12_recrit"] = params_instance.cxcl12_recrit
    ans["cxcl13_crit"] = params_instance.cxcl13_crit
    ans["cxcl13_diff_rate"] = params_instance.cxcl13_diff_rate
    ans["cxcl13_recrit"] = params_instance.cxcl13_recrit
    ans["dark_zone"] = params_instance.dark_zone
    ans["delete_ag_in_fresh_cc"] = params_instance.delete_ag_in_fresh_cc
    ans["dendrite_length"] = params_instance.dendrite_length
    ans["dif_delay"] = params_instance.dif_delay
    ans["dt"] = params_instance.dt
    ans["dx"] = params_instance.dx
    ans["exp_max"] = params_instance.exp_max
    ans["exp_min"] = params_instance.exp_min
    ans["initial_antigen_amount_per_fdc"] = params_instance.initial_antigen_amount_per_fdc
    ans["initial_num_fdc"] = params_instance.initial_num_fdc
    ans["initial_num_seeder"] = params_instance.initial_num_seeder
    ans["initial_num_stromal_cells"] = params_instance.initial_num_stromal_cells
    ans["initial_num_tcells"] = params_instance.initial_num_tcells
    ans["k_on"] = params_instance.k_on
    ans["light_zone"] = params_instance.light_zone
    ans["mutation_start_time"] = params_instance.mutation_start_time
    ans["n"] = params_instance.n
    ans["n_gc"] = params_instance.n_gc
    ans["north_weight"] = params_instance.north_weight
    ans["num_div_initial_cells"] = params_instance.num_div_initial_cells
    ans["offset"] = params_instance.offset
    ans["p_mhc_dep_max"] = params_instance.p_mhc_dep_max
    ans["p_mhc_dep_min"] = params_instance.p_mhc_dep_min
    ans["p_mhc_dep_nhill"] = params_instance.p_mhc_dep_nhill
    ans["p_mhc_depk"] = params_instance.p_mhc_depk
    ans["p_mk_cxcl12"] = params_instance.p_mk_cxcl12
    ans["p_mk_cxcl13"] = params_instance.p_mk_cxcl13
    ans["p_sel"] = params_instance.p_sel
    ans["plt_centroblast"] = params_instance.plt_centroblast
    ans["plt_centrocyte"] = params_instance.plt_centrocyte
    ans["plt_outcell"] = params_instance.plt_outcell
    ans["plt_tcell"] = params_instance.plt_tcell
    ans["pm_differentiation_rate"] = params_instance.pm_differentiation_rate
    ans["polarity_index"] = params_instance.polarity_index
    ans["possible_neighbours"] = params_instance.possible_neighbours
    ans["prob_dif"] = params_instance.prob_dif
    ans["prob_dif_to_out"] = params_instance.prob_dif_to_out
    ans["prob_divide_ag_asymmetric"] = params_instance.prob_divide_ag_asymmetric
    ans["prob_mut_affinity_exponent"] = params_instance.prob_mut_affinity_exponent
    ans["prob_mut_after_selection"] = params_instance.prob_mut_after_selection
    ans["prob_now"] = params_instance.prob_now
    ans["speed_centroblast"] = params_instance.speed_centroblast
    ans["speed_centrocyte"] = params_instance.speed_centrocyte
    ans["speed_outcell"] = params_instance.speed_outcell
    ans["speed_tcell"] = params_instance.speed_tcell
    ans["start_differentiation"] = params_instance.start_differentiation
    ans["tc_rescue_time"] = params_instance.tc_rescue_time
    ans["tc_time"] = params_instance.tc_time
    ans["test_delay"] = params_instance.test_delay
    ans["tmax"] = params_instance.tmax
    ans["tmin"] = params_instance.tmin
    ans["v_blood"] = params_instance.v_blood

    # Convert "north" from numpy array to tuple
    ans["north"] = params_instance.north.tolist()
    return ans


def dict_to_json(dictionary, filename):
    """
    Converts dictionary object to json file and saves.
    :param dictionary: dict, the dictionary we intended on saving
    :param filename: str, directory to save file at
    """

    with open(filename + ".json", 'w') as fp:
        json.dump(dictionary, fp, sort_keys=True)


def json_to_params(parameters, filename):
    """
    Modifies a Params object to contain the exact parameters
    wanted for current simulation.
    :param parameters: Params object, parameters to be over written
    :param filename: str, directory of saved json file containing parameters
    :return:
    """
    # Open json file
    with open(filename + ".json") as fp:
        params_dict = json.load(fp)

    # Modify current parameters object to be wanted values
    parameters.ab_prod_factor = params_dict["ab_prod_factor"]
    parameters.all_points = params_dict["all_points"]
    parameters.antibodies_production = params_dict["antibodies_production"]
    parameters.antibody_degradation = params_dict["antibody_degradation"]
    parameters.antigen_saturation = params_dict["antigen_saturation"]
    parameters.antigen_value = params_dict["antigen_value"]
    parameters.bcr_values_initial = params_dict["bcr_values_initial"]
    parameters.chemo_half = params_dict["chemo_half"]
    parameters.chemo_max = params_dict["chemo_max"]
    parameters.chemo_steep = params_dict["chemo_steep"]
    parameters.collect_fdc_period = params_dict["collect_fdc_period"]
    parameters.cxcl12_crit = params_dict["cxcl12_crit"]
    parameters.cxcl12_recrit = params_dict["cxcl12_recrit"]
    parameters.cxcl13_crit = params_dict["cxcl13_crit"]
    parameters.cxcl13_diff_rate = params_dict["cxcl13_diff_rate"]
    parameters.cxcl13_recrit = params_dict["cxcl13_recrit"]
    parameters.dark_zone = params_dict["dark_zone"]
    parameters.delete_ag_in_fresh_cc = params_dict["delete_ag_in_fresh_cc"]
    parameters.dendrite_length = params_dict["dendrite_length"]
    parameters.dif_delay = params_dict["dif_delay"]
    parameters.dt = params_dict["dt"]
    parameters.dx = params_dict["dx"]
    parameters.exp_max = params_dict["exp_max"]
    parameters.exp_min = params_dict["exp_min"]
    parameters.initial_antigen_amount_per_fdc = params_dict["initial_antigen_amount_per_fdc"]
    parameters.initial_num_fdc = params_dict["initial_num_fdc"]
    parameters.initial_num_seeder = params_dict["initial_num_seeder"]
    parameters.initial_num_stromal_cells = params_dict["initial_num_stromal_cells"]
    parameters.initial_num_tcells = params_dict["initial_num_tcells"]
    parameters.k_on = params_dict["k_on"]
    parameters.light_zone = params_dict["light_zone"]
    parameters.mutation_start_time = params_dict["mutation_start_time"]
    parameters.n = params_dict["n"]
    parameters.n_gc = params_dict["n_gc"]
    parameters.north_weight = params_dict["north_weight"]
    parameters.num_div_initial_cells = params_dict["num_div_initial_cells"]
    parameters.offset = params_dict["offset"]
    parameters.p_mhc_dep_max = params_dict["p_mhc_dep_max"]
    parameters.p_mhc_dep_min = params_dict["p_mhc_dep_min"]
    parameters.p_mhc_dep_nhill = params_dict["p_mhc_dep_nhill"]
    parameters.p_mhc_depk = params_dict["p_mhc_depk"]
    parameters.p_mk_cxcl12 = params_dict["p_mk_cxcl12"]
    parameters.p_mk_cxcl13 = params_dict["p_mk_cxcl13"]
    parameters.p_sel = params_dict["p_sel"]
    parameters.plt_centroblast = params_dict["plt_centroblast"]
    parameters.plt_centrocyte = params_dict["plt_centrocyte"]
    parameters.plt_outcell = params_dict["plt_outcell"]
    parameters.plt_tcell = params_dict["plt_tcell"]
    parameters.pm_differentiation_rate = params_dict["pm_differentiation_rate"]
    parameters.polarity_index = params_dict["polarity_index"]
    parameters.possible_neighbours = params_dict["possible_neighbours"]
    parameters.prob_dif = params_dict["prob_dif"]
    parameters.prob_dif_to_out = params_dict["prob_dif_to_out"]
    parameters.prob_divide_ag_asymmetric = params_dict["prob_divide_ag_asymmetric"]
    parameters.prob_mut_affinity_exponent = params_dict["prob_mut_affinity_exponent"]
    parameters.prob_mut_after_selection = params_dict["prob_mut_after_selection"]
    parameters.prob_now = params_dict["prob_now"]
    parameters.speed_centroblast = params_dict["speed_centroblast"]
    parameters.speed_centrocyte = params_dict["speed_centrocyte"]
    parameters.speed_outcell = params_dict["speed_outcell"]
    parameters.speed_tcell = params_dict["speed_tcell"]
    parameters.start_differentiation = params_dict["start_differentiation"]
    parameters.tc_rescue_time = params_dict["tc_rescue_time"]
    parameters.tc_time = params_dict["tc_time"]
    parameters.test_delay = params_dict["test_delay"]
    parameters.tmax = params_dict["tmax"]
    parameters.tmin = params_dict["tmin"]
    parameters.v_blood = params_dict["v_blood"]

    # Convert to numpy array
    parameters.north = np.array(params_dict["north"])


def start_out_csv(filename):
    """
    Creates csv file to store output. Writes the variable name for each column.
    :param filename: str, filename (and location) for csv file.
    :return:
    """
    variable_names = ["num_bcells",
                      "times"]

    with open(filename + ".csv", "w") as output_file:
        output_writer = csv.writer(output_file)
        output_writer.writerow(variable_names)


def update_out_csv(out_instance, filename):
    """
    Writes the current output values to csv file.
    :param out_instance: Out object, instance of all changing variables in simulation.
    :param filename: str, filename (and location) for csv file.
    :return:
    """
    new_line = [out_instance.times, out_instance.num_bcells]

    with open(filename + ".csv", "a") as output_file:
        output_writer = csv.writer(output_file)
        output_writer.writerow(new_line)


def pickle_current_state(out_instance, simulation_name):
    """
    Saves current state of simulation using pickle for the purpose of restarting simulation.
    :param out_instance: Out object, instance of all changing variables in simulation.
    :return:
    """
    restart_data = open(simulation_name + "_Restart_data{:04d}.pickle".format(out_instance.save_counter), "wb")
    pickle.dump(out_instance, restart_data)
    restart_data.close()


def recover_state_from_pickle(filename):
    """
    Searches through current directory to find most recent
    :return: Out object from simulation.
    """

    parameters_file = open(filename, "rb")
    output = pickle.load(parameters_file)
    return output


if __name__ == "__main__":
    """
    Requires one input to run the simulation. This input is the simulation name. 
    This will be used in the naming of the parameters json file, restarting pickle
    data and the output csv file.
    """
    assert len(sys.argv) == 2, "wrong number arguments given: {}".format(len(sys.argv))

    simulation_name = sys.argv[1]

    # Find all files in current directory
    all_files = os.listdir(".")

    # Generate Params object that might be overwritten with new values
    parameters = Params()

    # Check if files exist, if not, make them:
    if simulation_name + ".json" in all_files:
        json_to_params(parameters, simulation_name)
    else:
        parameters_dict = params_to_dict(parameters)
        dict_to_json(parameters_dict, simulation_name)

    # Find restart files and sort so latest in final position
    restart_files = [file for file in all_files if simulation_name + "_Restart_data" in file]
    restart_files.sort()
    if restart_files:
        # If restart files exist, load data
        output = recover_state_from_pickle(restart_files[-1])
    else:
        # otherwise, create new Out object
        output = Out(parameters)

    # If output file does not exist, create new one
    if simulation_name + ".csv" not in all_files:
        start_out_csv(simulation_name)

    # Starts/continues simulation
    hyphasma(parameters, output, simulation_name)
    plt.plot(output.times, output.num_bcells)
    plt.show()
