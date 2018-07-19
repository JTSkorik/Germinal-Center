# Imports:
import random
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
from enum import Enum


# TODO need to update all docstrings to reflect passing of varaibles
# TODO signal_secretion, diffusion, and turn_angle functions.

def mutate(id, parameters):
    """
    Algorithm 1, Mutation.
    Mutates the BCR value for a cell with a certain probability.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    # Determine where the BCR value should mutate.
    if random.uniform(0, 1) < parameters.pMutation[id]:
        # Choose index and value of BCR to change.
        index = random.choice([0, 1, 2, 3])
        value = int(str(parameters.BCR[id])[3 - index])
        # Randomly apply plus or minus one to one BCR position.
        if index == 3 and value == 1:
            parameters.BCR[id] += 10 ** index
        else:
            if value == 0:
                parameters.BCR[id] += 10 ** index
            elif value == 9:
                parameters.BCR[id] -= 10 ** index
            else:
                if random.uniform(0, 1) < 0.5:
                    parameters.BCR[id] += 10 ** index
                else:
                    parameters.BCR[id] -= 10 ** index
    # If new BCR value obtained, we need to start tracking its stats.
    if parameters.BCR[id] not in parameters.BCR_values_all:
        parameters.BCR_values_all.add(parameters.BCR[id])
        parameters.NumBCROutCells[parameters.BCR[id]] = 0
        parameters.NumBCROutCellsProduce[parameters.BCR[id]] = 0
        parameters.AntibodyPerBCR[parameters.BCR[id]] = 0


def initiate_chemokine_receptors(id, parameters):
    """
    Algorithm 2, Dynamic Updating of Chemotaxis.
    Used to determine the attributes responsive_to_CXCL12 and responsive_to_CXCL13 of a cell given its type.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    if parameters.Type[id] == CellType.Centroblast:
        parameters.responsiveToCXCL12[id] = True
        parameters.responsiveToCXCL13[id] = False
    elif parameters.Type[id] == CellType.Centrocyte:
        parameters.responsiveToCXCL12[id] = False
        parameters.responsiveToCXCL13[id] = True
    elif parameters.Type[id] == CellType.OutCell:
        parameters.responsiveToCXCL12[id] = False
        parameters.responsiveToCXCL13[id] = False
    else:
        print("initiate_chemokine_receptors: Invalid cell_type, {}".format(parameters.Type[id]))


def update_chemokines_receptors(id, parameters):
    """
    Algorithm 2, Dynamic Updating of Chemotaxis.
    Used to update the attributes responsive_to_CXCL12 and responsive_to_CXCL13 of a cell given its type and state.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    pos = parameters.Position[id]
    if parameters.Type[id] == CellType.Centrocyte:
        if parameters.State[id] in [CellState.Unselected, CellState.FDCselected]:
            if parameters.grid_cxcl13[pos] > parameters.CXCL13_CRIT:
                parameters.responsiveToCXCL13[id] = False
            elif parameters.grid_cxcl13[pos] < parameters.CXCL13_RECRIT:
                parameters.responsiveToCXCL13[id] = True
    elif parameters.Type[id] == CellType.Centroblast:
        if parameters.grid_cxcl12[pos] > parameters.CXCL12_CRIT:
            parameters.responsiveToCXCL12[id] = False
        elif parameters.grid_cxcl12[pos] < parameters.CXCL12_RECRIT:
            parameters.responsiveToCXCL12[id] = True


def move(id, parameters):
    # TODO add check that the cells are staying inside their respective zones.
    """
    Algorithm 3, Updating Position and Polarity of cells at each time-point.
    Updates the polarity of a cell and then will move the cell within the Germinal center. Both events occur stochastically.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    # Obtain current position of cell.
    pos = parameters.Position[id]
    x, y, z = pos

    # Obtain required parameters
    cell_type = parameters.Type[id]
    if cell_type == CellType.Centrocyte:
        prob = parameters.PLT_CENTROCYTE
        speed = parameters.SPEED_CENTROCYTE
    elif cell_type == CellType.Centroblast:
        prob = parameters.PLT_CENTROBLAST
        speed = parameters.SPEED_CENTROBLAST
    elif cell_type == CellType.TCell:
        prob = parameters.PLT_TCELL
        speed = parameters.SPEED_TCELL
    elif cell_type == CellType.OutCell:
        prob = parameters.PLT_OUTCELL
        speed = parameters.SPEED_OUTCELL
    else:
        prob = None
        speed = None
        print("move: Invalid cell_type, {}".format(cell_type))

    # Calculate new polarity
    if random.uniform(0, 1) < prob:
        # Turning angles influence
        theta = random.gauss(0, 1)
        phi = random.uniform(0, 2 * math.pi)
        turn_angle(id, theta, phi, parameters)

        # Influence due to CXCL12
        if parameters.responsiveToCXCL12[id]:
            x_diff = parameters.grid_cxcl12[(x + 1, y, z)] - parameters.grid_cxcl12[(x - 1, y, z)]
            y_diff = parameters.grid_cxcl12[(x, y + 1, z)] - parameters.grid_cxcl12[(x, y - 1, z)]
            z_diff = parameters.grid_cxcl12[(x, y, z + 1)] - parameters.grid_cxcl12[(x, y, z - 1)]

            Gradient_CXCL12 = np.array(
                [x_diff / (2 * parameters.DX), y_diff / (2 * parameters.DX), z_diff / (2 * parameters.DX)])

            mag_Gradient_CXCL12 = np.linalg.norm(Gradient_CXCL12)
            chemoFactor = (parameters.CHEMO_MAX / (
                1 + math.exp(parameters.CHEMO_STEEP * (
                    parameters.CHEMO_HALF - 2 * parameters.DX * mag_Gradient_CXCL12)))) * Gradient_CXCL12

            parameters.Polarity[id] += chemoFactor

        # Influence due to CXCL13
        if parameters.responsiveToCXCL13[id]:
            x_diff = parameters.grid_cxcl13[(x + 1, y, z)] - parameters.grid_cxcl13[(x - 1, y, z)]
            y_diff = parameters.grid_cxcl13[(x, y + 1, z)] - parameters.grid_cxcl13[(x, y - 1, z)]
            z_diff = parameters.grid_cxcl13[(x, y, z + 1)] - parameters.grid_cxcl13[(x, y, z - 1)]

            Gradient_CXCL13 = np.array(
                [x_diff / (2 * parameters.DX), y_diff / (2 * parameters.DX), z_diff / (2 * parameters.DX)])
            mag_Gradient_CXCL13 = np.linalg.norm(Gradient_CXCL13)
            chemoFactor = (parameters.CHEMO_MAX / (
                1 + math.exp(parameters.CHEMO_STEEP * (
                    parameters.CHEMO_HALF - 2 * parameters.DX * mag_Gradient_CXCL13)))) * Gradient_CXCL13

            parameters.Polarity[id] += chemoFactor

        # Influence specific to T Cells
        if cell_type == CellType.TCell:
            parameters.Polarity[id] = ((1.0 - parameters.NORTH_WEIGHT) * parameters.Polarity[id]) + (
                parameters.NORTH * parameters.NORTH_WEIGHT)

        parameters.Polarity[id] = parameters.Polarity[id] / np.linalg.norm(parameters.Polarity[id])

    # Probability of movement
    pDifu = speed * parameters.DT / parameters.DX

    if random.uniform(0, 1) < pDifu:
        # Find possible new positions based in order of best preference
        WantedPosition = np.asarray(pos) + parameters.Polarity[id]
        Neighbours = [np.asarray(Movement) + np.asarray(pos) for Movement in
                      parameters.POSSIBLE_NEIGHBOURS if np.linalg.norm(
                np.asarray(Movement) + np.asarray(pos) - np.array(parameters.OFFSET)) <= (parameters.N / 2)]
        Neighbours.sort(key=lambda w: np.linalg.norm(w - WantedPosition))

        # Move the cell to best available position that isn't against direction of polarity
        count = 0
        moved = False
        while not moved and count <= 9:
            new_pos = tuple(Neighbours[count])
            if parameters.Grid_ID[new_pos] is None:
                parameters.Position[id] = new_pos

                parameters.Grid_ID[new_pos] = parameters.Grid_ID[pos]
                parameters.Grid_Type[new_pos] = parameters.Grid_Type[pos]
                parameters.Grid_ID[pos] = None
                parameters.Grid_Type[pos] = None

                moved = True
            count += 1


def turn_angle(id, theta, phi, paramaters):
    """
    Algorithm 3 (Updating Position and Polarity of cells at each time-point)
    Rotates the polarity of a cell by the given angles.
    Yet to be finished.
    :param id: integer, determines which cell in the pop we are talking about.
    :param theta: float, randomly generated turning angle.
    :param phi: float, another randomly generated turning angle between 0 and 2pi.
    :return:
    """
    v = np.random.standard_normal(3)
    parameters.Polarity[id] = v / np.linalg.norm(v)


def initiate_cycle(id, parameters):
    """
    Algorithm 3, Updating events at the Centroblast Stage.
    Sets the state for a cell depending on how many divisions left to complete.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    # noinspection PyUnresolvedReferences
    if parameters.numDivisionsToDo[id] == 0:
        parameters.State[id] = CellState.Stop_Dividing
    else:
        parameters.State[id] = CellState.cb_G1
        parameters.cycleStartTime[id] = 0
        parameters.endOfThisPhase[id] = get_duration(parameters.State[id])


def progress_cycle(id, parameters):
    """
    Algorithm 4, Updating events at the Centroblast Stage.
    Progresses the cell to its next state and calculates how long it till next change of state.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    # Progress cell into its next state
    parameters.cycleStartTime[id] += parameters.DT
    if parameters.cycleStartTime[id] > parameters.endOfThisPhase[id]:
        if parameters.State[id] == CellState.cb_G1:
            parameters.State[id] = CellState.cb_S
        elif parameters.State[id] == CellState.cb_S:
            parameters.State[id] = CellState.cb_G2
        elif parameters.State[id] == CellState.cb_G2:
            parameters.State[id] = CellState.cb_M
        elif parameters.State[id] == CellState.cb_M:
            parameters.State[id] = CellState.cb_divide

        # Finds time till end of new state.
        if parameters.State[id] not in [CellState.cb_divide, CellState.Stop_Dividing]:
            parameters.endOfThisPhase[id] = get_duration(parameters.State[id])
            parameters.cycleStartTime[id] = 0


def divide_and_mutate(id, parameters):
    # TODO make sure cell doesnt divide into wrong zone
    """
    Algorithm 4, Updating events at the Centroblast Stage.
    Takes a cell and divides it with probability pNow. Will also attempt to mutate the cells.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    if random.uniform(0, 1) < parameters.PROB_NOW:
        # Find all empty positions neighbouring cell.
        pos = parameters.Position[id]
        empty_neighbours = [tuple(np.array(pos) + np.array(possible_neighbour)) for possible_neighbour in
                            parameters.POSSIBLE_NEIGHBOURS if np.linalg.norm(
                np.asarray(possible_neighbour) + np.asarray(pos) - np.array(parameters.OFFSET)) <= (parameters.N / 2)]

        # Randomly choose one position for new cell
        if empty_neighbours:
            divide_pos = random.choice(empty_neighbours)

            # Generate a new ID for the cell and copy over the properties from the old cell.
            new_id = parameters.available_cell_ids.pop()
            parameters.list_cb.append(new_id)

            parameters.Type[new_id] = CellType.Centroblast
            parameters.Position[new_id] = divide_pos
            parameters.State[new_id] = None
            parameters.BCR[new_id] = parameters.BCR[id]
            parameters.Polarity[new_id] = np.array(parameters.Polarity[id])
            parameters.responsiveToCXCL12[new_id] = True
            parameters.responsiveToCXCL13[new_id] = False
            parameters.numDivisionsToDo[new_id] = parameters.numDivisionsToDo[id] - 1
            parameters.pMutation[new_id] = parameters.pMutation[id]
            parameters.IAmHighAg[new_id] = False
            parameters.retainedAg[new_id] = None
            parameters.cycleStartTime[new_id] = None
            parameters.endOfThisPhase[new_id] = None

            parameters.numDivisionsToDo[id] -= 1
            parameters.IAmHighAg[id] = False

            parameters.Grid_ID[divide_pos] = new_id
            parameters.Grid_Type[divide_pos] = parameters.Type[new_id]

            # Initiate the cycle for each cell.
            initiate_cycle(id, parameters)
            initiate_cycle(new_id, parameters)

            # Mutate the cells
            if parameters.t > parameters.MUTATION_START_TIME:
                mutate(id, parameters)
                mutate(new_id, parameters)

            # Assign amount of retained antigen to each cell
            if random.uniform(0, 1) < parameters.PROB_DIVIDE_AG_ASYMMETRIC:
                if parameters.retainedAg[id] == 0:
                    parameters.retainedAg[new_id] = 0
                else:
                    sep = random.gauss(parameters.POLARITY_INDEX, 1)
                    while sep < 0 or sep > 1:
                        sep = random.gauss(parameters.POLARITY_INDEX, 1)

                    parameters.retainedAg[new_id] = sep * parameters.retainedAg[id]
                    parameters.retainedAg[id] = (1 - sep) * parameters.retainedAg[id]

                    if sep > 0.5:
                        parameters.IAmHighAg[new_id] = True
                    else:
                        parameters.IAmHighAg[id] = True
            else:
                parameters.retainedAg[id] = parameters.retainedAg[id] / 2
                parameters.retainedAg[new_id] = parameters.retainedAg[id]


def progress_fdc_selection(id, parameters):
    """
    Algorithm 5, Antigen Collection from FDCs.
    Allows for centrocytes to collect antigen from neighbouring F cells / fragments.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    if parameters.State[id] == CellState.Unselected:
        # Progress selected clock and check if able to collect antigen.
        parameters.selectedClock[id] += parameters.DT
        if parameters.selectedClock[id] <= parameters.COLLECT_FDC_PERIOD:
            parameters.Clock[id] += parameters.DT
            # noinspection PyTypeChecker
            if parameters.Clock[id] > parameters.TEST_DELAY:
                parameters.selectable[id] = True
                # Find frag component with largest amount of antigen.
                frag_max = None
                frag_max_id = None
                pos = parameters.Position[id]
                for neighbour in parameters.POSSIBLE_NEIGHBOURS:
                    neighbour_pos = tuple(np.array(pos) + np.array(neighbour))
                    if parameters.Grid_Type[neighbour_pos] in [CellType.Fragment, CellType.FCell]:
                        frag_id = parameters.Grid_ID[neighbour_pos]
                        if parameters.antigenAmount[frag_id] > frag_max:
                            frag_max = parameters.antigenAmount[frag_id]
                            frag_max_id = frag_id

                pBind = affinity(parameters.BCR[id]) * frag_max / parameters.ANTIGEN_SATURATION
                # Bind cell and fragment with probability pBind or reset clock.
                if random.uniform(0, 1) < pBind:
                    parameters.State[id] = CellState.FDCcontact
                    parameters.FragContact[id] = frag_max_id
                else:
                    parameters.Clock[id] = 0
                    parameters.selectable[id] = False
        else:
            # Cell dies if it doesn't get any contacts.
            if parameters.numFDCContacts[id] == 0:
                parameters.State[id] = CellState.Apoptosis
            else:
                parameters.State[id] = CellState.FDCselected

    elif parameters.State[id] == CellState.Contact:
        parameters.selectedClock[id] += parameters.DT
        if random.uniform(0, 1) < parameters.P_SEL:
            parameters.numFDCContacts[id] += 1
            frag_id = parameters.FragContact[id]
            parameters.antigenAmount[frag_id] -= 1
            parameters.State[id] = CellState.Unselected
            parameters.Clock[id] = 0
            parameters.selectable[id] = False


# noinspection PyTypeChecker,PyTypeChecker,PyTypeChecker
def progress_tcell_selection(id, parameters):
    """
    Algorithm 6, Screening for T cell help at Centrocyte Stage.

    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    if parameters.State[id] == CellState.FDCselected:
        # Find if there is a neighbouring T cell.
        pos = parameters.Position[id]
        for neighbour in parameters.POSSIBLE_NEIGHBOURS:
            neighbour_pos = tuple(np.array(pos) + np.array(neighbour))
            # If there is a neighbouring T cell, we record the contact.
            if parameters.Grid_Type[neighbour_pos] == CellType.TCell and parameters.State[id] != CellState.TCcontact:
                update_tcell(id, parameters.Grid_ID[neighbour_pos], parameters)
                parameters.State[id] = CellState.TCcontact
                parameters.tcClock[id] = 0
                parameters.tcSignalDuration[id] = 0

    elif parameters.State[id] == CellState.TCcontact:
        parameters.tcClock[id] += parameters.DT
        tcell_id = parameters.TCell_Contact[id]
        # Check is current cell has least amount of antigens compared to T cells neighbouring cells.
        lowest_antigen = True
        for id_bc in parameters.BCell_Contacts[tcell_id]:
            if id != id_bc and parameters.retainedAg[id] <= parameters.retainedAg[id_bc]:
                lowest_antigen = False

        if lowest_antigen:
            parameters.tcSignalDuration[id] += parameters.DT
        if parameters.tcSignalDuration[id] > parameters.TC_RESCUE_TIME:
            parameters.State[id] = CellState.Selected
            parameters.selectedClock[id] = 0
            rand = random.uniform(0, 1)
            parameters.IndividualDifDelay[id] = parameters.DIF_DELAY * (1 + 0.1 * math.log(1 - rand) / rand)
            liberate_tcell(id, parameters.TCell_Contact[id], parameters)
        elif parameters.tcClock[id] >= parameters.TC_TIME:
            parameters.State[id] = CellState.Apoptosis
            liberate_tcell(id, parameters.TCell_Contact[id], parameters)

    elif parameters.State[id] == CellState.Selected:
        parameters.selectedClock[id] += parameters.DT
        if parameters.selectedClock[id] > parameters.IndividualDifDelay[id]:
            if random.uniform(0, 1) < parameters.PROB_DIF:
                if random.uniform(0, 1) < parameters.PROB_DIF_TO_OUT:
                    differ_to_out(id, parameters)
                else:
                    differ_to_cb(id, parameters)


def update_tcell(id_b, id_t, parameters):
    """
    Algorithm 7, Updating the T cells according to B cells interactions.
    Updates records of whether cells are interacting, specifically marking two cells as interacting.
    :param id_b: integer, determines which B cell in the pop we are talking about.
    :param id_t: integer, determines which T cell in the pop we are talking about.
    :return:
    """

    parameters.BCell_Contacts[id_t].append(id_b)
    parameters.TCell_Contact[id_b] = id_t
    parameters.State[id_t] = CellState.TC_CC_Contact


def liberate_tcell(id_b, id_t, parameters):
    """
    Algorithm 7, Updating the T cells according to B cells interactions.
    Updates records of whether cells are interacting, specifically removing cells from being interactive.
    :param id_b: integer, determines which B cell in the pop we are talking about.
    :param id_t: integer, determines which T cell in the pop we are talking about.
    :return:
    """

    parameters.BCell_Contacts[id_t].remove(id_b)
    if not parameters.BCell_Contacts[id_t]:
        parameters.State[id_t] = CellState.TCnormal
    parameters.TCell_Contact[id_b] = None


def differ_to_out(id, parameters):
    """
    Algorithm 8, Transition between Centroblasts, Centrocyctes, and Output Cells.
    Transition a cell from Centroblast or Centrocycte to an Output Cell.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    parameters.list_outcells.append(id)
    parameters.NumBCROutCells[parameters.BCR[id]] += 1
    # Update cell and Grid position properties.
    parameters.Type[id] = CellType.OutCell
    parameters.responsiveToCXCL12[id] = None
    parameters.responsiveToCXCL13[id] = None
    parameters.Grid_Type[parameters.Position[id]] = CellType.OutCell
    initiate_chemokine_receptors(id, parameters)


def differ_to_cb(id, parameters):
    """
    Algorithm 8, Transition between Centroblasts, Centrocyctes, and Output Cells.
    Transitions a cell from Centrocyte to Centroblast.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    parameters.list_cb.append(id)

    # Update grid and cell details.
    parameters.Grid_Type[parameters.Position[id]] = CellType.Centroblast

    parameters.Type[id] = CellType.Centroblast
    parameters.responsiveToCXCL12[id] = None
    parameters.responsiveToCXCL13[id] = None
    parameters.numDivisionsToDo[id] = None
    parameters.pMutation[id] = None
    parameters.IAmHighAg[id] = True
    parameters.retainedAg[id] = parameters.NumFDCContacts[id]
    parameters.cycleStartTime[id] = None
    parameters.endOfThisPhase[id] = None

    # Find number of divisions to do.
    agFactor = parameters.numFDCContacts[id] ** parameters.pMHCdepHill
    parameters.numDivisionsToDo[id] = parameters.pMHCdepMin + (
                                                                  parameters.pMHCdepMax - parameters.pMHCdepMin) * agFactor / (
                                                                  agFactor + parameters.pMHCdepK ** parameters.pMHCdepHill)

    # Find new probability of mutation.
    parameters.pMutation[id] = p_mut(parameters.t) + (parameters.PROB_MUT_AFTER_SELECTION - p_mut(
        parameters.t)) * affinity(
        parameters.BCR[id]) ** parameters.PROB_MUT_AFFINITY_EXPONENT

    # Initiate cell.
    initiate_chemokine_receptors(id, parameters)
    initiate_cycle(id, parameters)


def differ_to_cc(id, parameters):
    """
    Algorithm 8, Transition between Centroblasts, Centrocyctes, and Output Cells.
    Transitions a cell from Centroblast to Centrocyte.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    parameters.list_cc.append(id)
    old_retainedAg = parameters.retainedAg[id]

    parameters.Type[id] = CellType.Centrocyte
    parameters.State[id] = CellState.Unselected
    parameters.responsiveToCXCL12[id] = None
    parameters.responsiveToCXCL13[id] = None
    parameters.selectedClock[id] = 0.0
    parameters.Clock[id] = 0.0
    parameters.selectable[id] = False
    parameters.FragContact[id] = None
    parameters.numFDCContacts[id] = None
    parameters.tcClock[id] = None
    parameters.tcSignalDuration[id] = None
    parameters.IndividualDifDelay[id] = None
    parameters.TCell_Contact[id] = None

    initiate_chemokine_receptors(id, parameters)
    parameters.Grid_Type[parameters.Position[id]] = CellType.Centrocyte
    if parameters.DELETE_AG_IN_FRESH_CC:
        parameters.numFDCContacts[id] = 0.0
    else:
        parameters.numFDCContacts[id] = math.floor(old_retainedAg + 0.5)


def initialise_cells(parameters):
    """
    Algorithm 9, Initialisation.
    Starts the simulation by introducing various amounts of the possible cells into the simulation.
    :return:
    """

    # Initialise Stromal Cells:
    for _ in range(parameters.INITIAL_NUM_STROMAL_CELLS):
        # Find empty location in dark zone
        pos = random.choice(parameters.DARK_ZONE)
        while parameters.Grid_ID[pos] is not None:
            pos = random.choice(parameters.DARK_ZONE)

        # Obtain new ID for new cell
        id = parameters.available_cell_ids.pop()

        # Add to appropriate lists and dictionaries
        parameters.list_stromal.append(id)
        parameters.Type[id] = CellType.Stromal
        parameters.Position[id] = pos

        parameters.Grid_ID[pos] = id
        parameters.Grid_Type[pos] = CellType.Stromal

    # Initialise Fragments:
    for _ in range(parameters.INITIAL_NUM_FDC):
        # Find empty location in light zone
        pos = random.choice(parameters.LIGHT_ZONE)
        while parameters.Grid_ID[pos] is not None:
            pos = random.choice(parameters.LIGHT_ZONE)

        # Obtain new ID for new cell
        id = parameters.available_cell_ids.pop()

        # Add to appropriate lists and dictionaries
        parameters.list_fdc.append(id)

        parameters.Type[id] = CellType.FCell
        parameters.Position[id] = pos
        parameters.antigenAmount[id] = None
        parameters.icAmount[id] = 0
        parameters.Fragments[id] = []

        parameters.Grid_ID[pos] = id
        parameters.Grid_Type[pos] = CellType.FCell

        # Find the fragments for the FCell
        fcell_id = id
        fragments = parameters.Fragments[id]
        x, y, z = pos
        for i in range(1, parameters.DENDRITE_LENGTH + 1):
            for frag_pos in [(x + i, y, z), (x - i, y, z), (x, y + i, z), (x, y - i, z), (x, y, z - i)]:
                if (frag_pos[0] - (parameters.N / 2 + 0.5)) ** 2 + (frag_pos[1] - (parameters.N / 2 + 0.5)) ** 2 + (
                            frag_pos[2] - (parameters.N / 2 + 0.5)) ** 2 <= (parameters.N / 2) ** 2 and \
                                parameters.Grid_ID[frag_pos] is None:
                    id = parameters.available_cell_ids.pop()
                    fragments.append(id)

                    parameters.Type[id] = CellType.Fragment
                    parameters.Position[id] = frag_pos
                    parameters.antigenAmount[id] = None
                    parameters.icAmount[id] = 0
                    parameters.Parent[id] = fcell_id

                    parameters.Grid_ID[frag_pos] = id
                    parameters.Grid_Type[frag_pos] = CellType.Fragment

            # When Z axis is increasing, we require an extra check to ensure that we're still in light zone.
            frag_pos = (x, y, z + i)
            if (frag_pos[0] - (parameters.N / 2 + 0.5)) ** 2 + (frag_pos[1] - (parameters.N / 2 + 0.5)) ** 2 + (
                        frag_pos[2] - (parameters.N / 2 + 0.5)) ** 2 <= (parameters.N / 2) ** 2 and frag_pos[
                2] <= parameters.N // 2 and parameters.Grid_ID[
                frag_pos] is None:
                id = parameters.available_cell_ids.pop()
                fragments.append(id)

                parameters.Type[id] = CellType.Fragment
                parameters.Position[id] = frag_pos
                parameters.antigenAmount[id] = None
                parameters.icAmount[id] = 0
                parameters.Parent[id] = fcell_id

                parameters.Grid_ID[frag_pos] = id
                parameters.Grid_Type[frag_pos] = CellType.Fragment

        # Assign each fragment an amount of antigen
        fcell_vol = len(fragments) + 1  # +1 accounts for centre
        ag_per_frag = parameters.INITIAL_ANTIGEN_AMOUNT_PER_FDC / fcell_vol
        for id in [fcell_id] + fragments:
            parameters.antigenAmount[id] = ag_per_frag

    # Initialise Seeder Cells:
    for _ in range(parameters.INITIAL_NUM_SEEDER):
        pos = random.choice(parameters.LIGHT_ZONE)  # Find empty location in list zone
        while parameters.Grid_ID[pos] is not None:
            pos = random.choice(parameters.LIGHT_ZONE)

        # Obtain new ID for new cell
        id = parameters.available_cell_ids.pop()

        # Add cell to appropriate lists and dictionaries
        parameters.list_cb.append(id)
        polarity_vector = np.random.standard_normal(3)
        polarity_vector = polarity_vector / np.linalg.norm(polarity_vector)

        parameters.Type[id] = CellType.Centroblast
        parameters.Position[id] = pos
        parameters.State[id] = None
        parameters.BCR[id] = random.choice(parameters.BCR_values_initial)
        parameters.Polarity[id] = polarity_vector
        parameters.responsiveToCXCL12[id] = None
        parameters.responsiveToCXCL13[id] = None
        parameters.numDivisionsToDo[id] = parameters.NUM_DIV_INITIAL_CELLS
        parameters.pMutation[id] = p_mut(parameters.t)
        parameters.IAmHighAg[id] = False
        parameters.retainedAg[id] = 0.0
        parameters.cycleStartTime[id] = None
        parameters.endOfThisPhase[id] = None

        initiate_cycle(id, parameters)
        initiate_chemokine_receptors(id, parameters)

    # Initialise T Cells:
    for _ in range(parameters.INITIAL_NUM_TC):
        pos = random.choice(parameters.LIGHT_ZONE)  # Find empty location in light zone
        while parameters.Grid_ID[pos] is not None:
            pos = random.choice(parameters.LIGHT_ZONE)

        # Obtain new ID for new cell.
        id = parameters.available_cell_ids.pop()

        # Add cell to appropriate lists and dictionaries
        parameters.list_tc.append(id)
        polarity_vector = np.random.standard_normal(3)
        polarity_vector = polarity_vector / np.linalg.norm(polarity_vector)

        parameters.Type[id] = CellType.TCell
        parameters.Position[id] = pos
        parameters.State[id] = CellState.TCnormal
        parameters.Polarity[id] = polarity_vector
        parameters.responsiveToCXCL12[id] = False
        parameters.responsiveToCXCL13[id] = False
        parameters.BCell_Contacts[id] = []

        parameters.Grid_ID[pos] = id
        parameters.Grid_Type[pos] = CellType.TCell


def hyphasma(parameters):
    """
    Algorithm 10, Simulation of Germinal Center.
    Main driver function for the simulation of a Germinal Center.
    :return:
    """

    initialise_cells(parameters)

    while parameters.t <= parameters.TMAX:
        # Some of the following lines are purely for checking the simulation is running alright.
        parameters.t = round(parameters.t, 3)
        print(parameters.t)
        # Track the number of B cellsat each time step.
        parameters.num_bcells.append(len(parameters.list_cc) + len(parameters.list_cb))
        if parameters.num_bcells[-1] > 3:
            print("Number B Cells: {}".format(parameters.num_bcells[-1]))
            print("Number Centroblasts: {}".format(len(parameters.list_cb)))
            print("Number Centrocytes: {}".format(len(parameters.list_cc)))
        parameters.times.append(parameters.t)

        # Secrete CXCL12 from Stromal Cells.
        for id in parameters.list_stromal:
            signal_secretion(id, 'CXCL12', parameters.p_mkCXCL12)

        # Randomly iterate over F cells / Fragments.
        random.shuffle(parameters.list_fdc)
        for id in parameters.list_fdc:
            # Secrete CXCL13 from Fcells
            signal_secretion(id, 'CXCL13', parameters.p_mkCXCL13)
            fragments = parameters.Fragments[id]
            # Update antigen amounts for each fragment.
            for frag_id in fragments:
                for bcr_seq in parameters.BCR_values_all:
                    d_ic = parameters.DT * (
                        parameters.K_ON * parameters.antigenAmount[frag_id] * parameters.AntibodyPerBCR[
                            bcr_seq] - k_off(
                            bcr_seq) * parameters.icAmount[frag_id])
                    parameters.antigenAmount[frag_id] -= d_ic
                    parameters.icAmount[frag_id] += d_ic

        # Update the number of outcells and amount of antibody for each BCR value.
        for bcr_seq in parameters.BCR_values_all:
            transfert = math.floor(
                parameters.NumBCROutCells[bcr_seq] * parameters.PM_DIFFERENTIATION_RATE * parameters.DT)
            parameters.NumBCROutCells[bcr_seq] -= transfert
            parameters.NumBCROutCellsProduce[bcr_seq] += transfert
            parameters.AntibodyPerBCR[bcr_seq] = parameters.NumBCROutCellsProduce[
                                                     bcr_seq] * parameters.AB_PROD_FACTOR - parameters.ANTIBODY_DEGRADATION * \
                                                                                            parameters.AntibodyPerBCR[
                                                                                                bcr_seq]
        # Randomly iterate over Outcells
        random.shuffle(parameters.list_outcells)
        for id in parameters.list_outcells:
            # Move cell and remove if on surface of sphere / Germinal Center.
            move(id, parameters)
            pos = parameters.Position[id]
            if is_surface_point(pos):
                parameters.list_outcells.remove(id)
                parameters.available_cell_ids.append(id)

        # Randomly iterate over Centroblast cells.
        random.shuffle(parameters.list_cb)
        for id in parameters.list_cb:
            # Progress cells in their lifetime cycle.
            update_chemokines_receptors(id, parameters)
            progress_cycle(id, parameters)
            # Attempt to divide cell if ready.
            if parameters.State[id] == CellState.cb_divide:
                divide_and_mutate(id, parameters)
            if parameters.State[id] == CellState.Stop_Dividing:
                if random.uniform(0, 1) < parameters.PROB_DIF:
                    if parameters.IAmHighAg[id]:
                        differ_to_out(id, parameters)
                        parameters.list_cb.remove(id)
                    else:
                        differ_to_cc(id, parameters)
                        parameters.list_cb.remove(id)

            # Move allowed cells.
            if parameters.State[id] != CellState.cb_M:
                move(id, parameters)

        # Randomly iterate over Centrocyte cells.
        random.shuffle(parameters.list_cc)
        for id in parameters.list_cc:
            # Update cells progress
            update_chemokines_receptors(id, parameters)
            progress_fdc_selection(id, parameters)
            progress_tcell_selection(id, parameters)
            # Remove cell from simulation if dead or move if not in contact with T or F cell.
            if parameters.State[id] == CellState.Apoptosis:
                parameters.list_cb.remove(id)
                parameters.available_cell_ids.append(id)
            elif parameters.State[id] not in [CellState.FDCcontact, CellState.TCcontact]:
                move(id, parameters)

        # Randomly iterate over T cells and move if not attached to another cell.
        random.shuffle(parameters.list_tc)
        for id in parameters.list_tc:
            if parameters.State[id] == CellState.TCnormal:
                move(id, parameters)

        # At this point we can add in more B cells using a similar process to algorithm 9.

        # TODO collision resolution, lines 68-72
        parameters.t += parameters.DT


def initialise_germinal_center(N):
    """
    Might use to initialise grid before initialising cells.
    :param N:
    :return:
    """
    pass


# Extra Algorithms/Functions

def affinity(bcr):
    """
    Calculates the affinity between the Antigen and a given cells BCR.
    :param : integer, determines which cell in the pop we are talking about.
    :return: float, affinity value.
    """
    # noinspection PyUnresolvedReferences
    hamming_dist = sum(el1 != el2 for el1, el2 in zip(str(bcr), str(ANTIGEN_VALUE)))
    return math.exp(-(hamming_dist / 2.8) ** 2)


def k_off(bcr):
    """
    Calulates k_off
    :param bcr: BCR value that has been exhibited throughout simulation.
    :return: float, value of k_off
    """
    hamming_dist = sum(el1 != el2 for el1, el2 in zip(str(bcr), str(parameters.ANTIGEN_VALUE)))
    return parameters.K_ON / (
        10 ** (parameters.EXP_MIN + math.exp(-(hamming_dist / 2.8) ** 2) * (parameters.EXP_MAX - parameters.EXP_MIN)))


def p_mut(t):
    """
    Finds the probability that a cell will mutate without extra influences.
    :param t: float, current time of the simulation.
    :return: float, probability of mutation.
    """
    if t > 24:
        return 0.5
    else:
        return 0.0


# noinspection PyUnresolvedReferences
def get_duration(state):
    """
    Find duration of time before a cell divides. Amount of time determined using Guassian Random Variable.
    :param state:
    :return: float, sample from a Guassian random variable representing time till a cell divides.
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
        print("getDuration: Invalid cell state, {}".format(cell.State))
    mu = mu / 3
    result = random.gauss(mu, sigma)
    while result < 0:
        result = random.gauss(mu, sigma)
    return result


def signal_secretion(ID, chem, chem_prod_rate):
    pass


def diffuse_signal():
    pass


def is_surface_point(position):
    """
    Determines if a position in the Germinal Center is on the surface. If the neighbour of a position in the Germinal
    Center is outside the simulation domain, we consider that position to be on the surface.
    :param position: 3-tuple, the x,y,z position of a cell.
    :return surface: boolean, Boolean response for whether the given point is within the Germinal Center.
    """
    pos = np.array(position)
    surface = False
    # Test the main neighbouring points to determine if they are inside the Germinal Center
    for movement in [np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, -1, 0]),
                     np.array([0, 0, 1]), np.array([0, 0, -1])]:
        neighbour_pos = pos + movement
        if (neighbour_pos[0] - (parameters.N / 2 + 0.5)) ** 2 + (neighbour_pos[1] - (parameters.N / 2 + 0.5)) ** 2 + (
                    neighbour_pos[2] - (parameters.N / 2 + 0.5)) ** 2 > (parameters.N / 2) ** 2:
            surface = True
    return surface


class Params():
    """
    Class to store all variables to be passed around to each function.
    """

    def __init__(self):
        # Set-up for simulation:
        self.ANTIGEN_VALUE = 1234

        # Distance Variables:
        self.N = 16  # Diameter of sphere/GC
        self.ALL_POINTS = [(x + self.N // 2 + 1, y + self.N // 2 + 1, z + self.N // 2 + 1) for x in
                           range(-self.N // 2, self.N // 2) for y in
                           range(-self.N // 2, self.N // 2)
                           for z in range(-self.N // 2, self.N // 2) if
                           ((x + 0.5) ** 2 + (y + 0.5) ** 2 + (z + 0.5) ** 2) <= (self.N // 2) ** 2]
        self.DARK_ZONE = [point for point in self.ALL_POINTS if point[2] > self.N // 2]
        self.LIGHT_ZONE = [point for point in self.ALL_POINTS if point[2] <= self.N // 2]
        self.OFFSET = (self.N / 2 + 0.5, self.N / 2 + 0.5, self.N / 2 + 0.5)

        # Spatial step size (micrometers)
        self.DX = 5

        # Time Variables:
        self.DT = 0.002
        self.t = 0.0
        self.TMIN = 0.0
        self.TMAX = 30.0

        # Available Cell IDs:
        self.available_cell_ids = list(range(len(self.ALL_POINTS)))

        # Lists to store ID of each cell in each state (and fragments)
        self.list_stromal = []
        self.list_fdc = []
        self.list_cb = []
        self.list_cc = []
        self.list_tc = []
        self.list_outcells = []

        # Initialisation
        self.INITIAL_NUM_STROMAL_CELLS = 30
        self.INITIAL_NUM_FDC = 20
        self.INITIAL_NUM_SEEDER = 3
        self.INITIAL_NUM_TC = 25

        self.DENDRITE_LENGTH = 8
        self.INITIAL_ANTIGEN_AMOUNT_PER_FDC = 3000

        # Possible initial BCR values
        self.BCR_values_initial = random.sample(range(1000, 10000), 1000)
        self.BCR_values_all = set(self.BCR_values_initial)
        self.NumBCROutCells = {bcr: 0 for bcr in self.BCR_values_initial}
        self.NumBCROutCellsProduce = {bcr: 0 for bcr in self.BCR_values_initial}
        self.AntibodyPerBCR = {bcr: 0 for bcr in self.BCR_values_initial}

        # Numpy Arrays storing what is at each location. Outside of sphere the points take value -1,
        # initially the points inside the sphere take value None.
        self.Grid_ID = np.full((self.N + 2, self.N + 2, self.N + 2), -1, dtype=object)
        self.Grid_Type = np.full((self.N + 2, self.N + 2, self.N + 2), -1, dtype=object)
        for point in self.ALL_POINTS:
            self.Grid_ID[point] = None
            self.Grid_Type[point] = None

        # Dictionaries storing amounts of CXCL12 and CXCL13 at each point:


        self.grid_cxcl12 = np.random.uniform(80e-11, 80e-10, (self.N + 2, self.N + 2, self.N + 2))
        self.grid_cxcl12 = np.random.uniform(0.1e-10, 0.1e-9, (self.N + 2, self.N + 2, self.N + 2))

        # Dynamic number of divisions:
        self.NUM_DIV_INITIAL_CELLS = 3
        self.pMHCdepHill = 1.0
        self.pMHCdepMin = 1.0
        self.pMHCdepMax = 6.0
        self.pMHCdepK = 6.0

        # Production/ Diffusion Rates:
        self.p_mkCXCL12 = 4e-7
        self.p_mkCXCL13 = 1e-8
        self.CXCL13_DiffRate = 1000 * 25 * 0.002

        # Persistent Length time (PLT)
        self.PLT_CENTROCYTE = 0.025
        self.PLT_CENTROBLAST = 0.025
        self.PLT_TCELL = 0.0283
        self.PLT_OUTCELL = 0.0125

        # Dynamic update of chemokine receptors
        self.CXCL12_CRIT = 60.0e-10
        self.CXCL12_RECRIT = 40.0e-10
        self.CXCL13_CRIT = 0.8e-10
        self.CXCL13_RECRIT = 0.6e-10

        # Chemotaxis
        self.CHEMO_MAX = 10
        self.CHEMO_STEEP = 1e+10
        self.CHEMO_HALF = 2e-11
        self.NORTH_WEIGHT = 0.1
        self.NORTH = np.array([0, 0, -1])

        # Speed
        self.SPEED_CENTROCYTE = 7.5
        self.SPEED_CENTROBLAST = 7.5
        self.SPEED_TCELL = 10.0
        self.SPEED_OUTCELL = 3.0

        # Divide and Mutate
        self.PROB_NOW = self.DT * 9.0 * 10
        self.MUTATION_START_TIME = 2.0
        self.POLARITY_INDEX = 0.88
        self.PROB_DIVIDE_AG_ASYMMETRIC = 0.72
        self.PROB_MUT_AFTER_SELECTION = 0.0
        self.PROB_MUT_AFFINITY_EXPONENT = 1.0

        # Differentiation Rates
        self.START_DIFFERENTIATION = 72.0
        self.PROB_DIF = self.DT * 0.1
        self.DELETE_AG_IN_FRESH_CC = True
        self.DIF_DELAY = 6.0

        self.PROB_DIF_TO_OUT = 0.0

        # Selection Steps
        self.TEST_DELAY = 0.02
        self.COLLECT_FDC_PERIOD = 0.7
        self.ANTIGEN_SATURATION = 20
        self.P_SEL = self.DT * 0.05
        self.TC_TIME = 0.6
        self.TC_RESCUE_TIME = 0.5

        # Antibody
        self.PM_DIFFERENTIATION_RATE = 24.0
        self.ANTIBODIES_PRODUCTION = 1e-17
        self.V_BLOOD = 1e-2
        self.N_GC = 1000
        self.AB_PROD_FACTOR = self.ANTIBODIES_PRODUCTION * self.DT * self.N_GC * self.V_BLOOD * 1e15
        self.ANTIBODY_DEGRADATION = 30
        self.K_ON = 3.6e9

        self.EXP_MIN = 5.5
        self.EXP_MAX = 9.5

        # Movements:
        self.POSSIBLE_NEIGHBOURS = list(itertools.product([-1, 0, 1], repeat=3))
        self.POSSIBLE_NEIGHBOURS.remove((0, 0, 0))

        # For plots/tests:
        self.num_bcells = []
        self.times = []

        # Cell Properties
        # General
        self.Type = {}
        self.Position = {}
        self.State = {}
        self.BCR = {}
        self.Polarity = {}
        self.responsiveToCXCL12 = {}
        self.responsiveToCXCL13 = {}

        # Centroblasts
        self.numDivisionsToDo = {}
        self.pMutation = {}
        self.IAmHighAg = {}
        self.retainedAg = {}
        self.cycleStartTime = {}
        self.endOfThisPhase = {}

        # Centrocytes
        self.selectedClock = {}
        self.Clock = {}
        self.selectable = {}
        self.FragContact = {}
        self.numFDCContacts = {}
        self.tcClock = {}
        self.tcSignalDuration = {}
        self.IndividualDifDelay = {}
        self.TCell_Contact = {}

        # F Cells & Fragments
        self.antigenAmount = {}
        self.icAmount = {}
        self.Fragments = {}
        self.Parent = {}

        # T Cells
        self.BCell_Contacts = {}


# Enumerations to replace string comparisons:

class CellType(Enum):
    Stromal = 1
    FCell = 2
    Fragment = 3
    TCell = 4
    Centroblast = 5
    Centrocyte = 6
    OutCell = 7


class CellState(Enum):
    # T cells
    TCnormal = 10
    TC_CC_Contact = 11

    # Centroblasts
    cb_G1 = 12
    cb_G0 = 13
    cb_S = 14
    cb_G2 = 15
    cb_M = 16
    Stop_Dividing = 17
    cb_divide = 18

    # Centrocytes
    Unselected = 19
    FDCcontact = 20
    FDCselected = 21
    TCcontact = 22
    Selected = 23
    Apoptosis = 24

    # Outcell
    OutCell = 25


# Run Simulation:
if __name__ == "__main__":
    parameters = Params()
    hyphasma(parameters)
    plt.plot(parameters.times, parameters.num_bcells)
    plt.show()
