# Imports:
import random
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
from enum import Enum
from types import SimpleNamespace
import copy

THIS_IS_A_GLOBAL_VALRIABLE = 3


# TODO need to update all docstrings to reflect passing of varaibles
# TODO signal_secretion, diffusion, and turn_angle functions.

def mutate(id, parameters):
    """
    Algorithm 1, Mutation.
    Mutates the BCR value for a cell with a certain probability.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    cell = parameters.cell_properties[id]
    # Determine where the BCR value should mutate.
    if random.uniform(0, 1) < cell.pMutation:
        # Choose index and value of BCR to change.
        index = random.choice([0, 1, 2, 3])
        value = int(str(cell.BCR)[3 - index])
        # Randomly apply plus or minus one to one BCR position.
        if index == 3 and value == 1:
            cell.BCR += 10 ** index
        else:
            if value == 0:
                cell.BCR += 10 ** index
            elif value == 9:
                cell.BCR -= 10 ** index
            else:
                if random.uniform(0, 1) < 0.5:
                    cell.BCR += 10 ** index
                else:
                    cell.BCR -= 10 ** index
    # If new BCR value obtained, we need to start tracking its stats.
    # TODO add these to inputs and return
    if cell.BCR not in parameters.BCR_values_all:
        parameters.BCR_values_all.add(cell.BCR)
        parameters.NumBCROutCells[cell.BCR] = 0
        parameters.NumBCROutCellsProduce[cell.BCR] = 0
        parameters.AntibodyPerBCR[cell.BCR] = 0


def initiate_chemokine_receptors(id, parameters):
    """
    Algorithm 2, Dynamic Updating of Chemotaxis.
    Used to determine the attributes responsive_to_CXCL12 and responsive_to_CXCL13 of a cell given its type.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    cell = parameters.cell_properties[id]
    if cell.Type == CellType.Centroblast:
        cell.responsiveToCXCL12 = True
        cell.responsiveToCXCL13 = False
    elif cell.Type == CellType.Centrocyte:
        cell.responsiveToCXCL12 = False
        cell.responsiveToCXCL13 = True
    elif cell.Type == CellType.OutCell:
        cell.responsiveToCXCL12 = False
        cell.responsiveToCXCL13 = True
    else:
        print("initiate_chemokine_receptors: Invalid cell_type, {}".format(cell.Type))


def update_chemokines_receptors(id, parameters):
    """
    Algorithm 2, Dynamic Updating of Chemotaxis.
    Used to update the attributes responsive_to_CXCL12 and responsive_to_CXCL13 of a cell given its type and state.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    cell = parameters.cell_properties[id]
    pos = cell.Position
    if cell.Type == CellType.Centrocyte:
        if cell.State == CellState.Unselected:
            if parameters.grid_cxcl13[pos] > parameters.CXCL13_CRIT:
                cell.responsiveToCXCL13 = False
            elif parameters.grid_cxcl13[pos] < parameters.CXCL13_RECRIT:
                cell.responsiveToCXCL13 = True
    elif cell.Type == CellType.Centroblast:
        if parameters.grid_cxcl12[pos] > parameters.CXCL12_CRIT:
            cell.responsiveToCXCL12 = False
        elif parameters.grid_cxcl12[pos] < parameters.CXCL12_RECRIT:
            cell.responsiveToCXCL12 = True


def move(id, parameters):
    # TODO add check that the cells are staying inside their respective zones.
    """
    Algorithm 3, Updating Position and Polarity of cells at each time-point.
    Updates the polarity of a cell and then will move the cell within the Germinal center. Both events occur stochastically.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    cell = parameters.cell_properties[id]
    # Obtain current position of cell.
    pos = cell.Position
    x, y, z = pos

    # Obtain required parameters
    cell_type = cell.Type
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
        turn_angle(id, theta, phi, parameters.cell_properties)

        # Influence due to CXCL12
        if cell.responsiveToCXCL12:
            x_diff = parameters.grid_cxcl12[(x + 1, y, z)] - parameters.grid_cxcl12[(x - 1, y, z)]
            y_diff = parameters.grid_cxcl12[(x, y + 1, z)] - parameters.grid_cxcl12[(x, y - 1, z)]
            z_diff = parameters.grid_cxcl12[(x, y, z + 1)] - parameters.grid_cxcl12[(x, y, z - 1)]

            Gradient_CXCL12 = np.array(
                [x_diff / (2 * parameters.DX), y_diff / (2 * parameters.DX), z_diff / (2 * parameters.DX)])

            mag_Gradient_CXCL12 = np.linalg.norm(Gradient_CXCL12)
            chemoFactor = (parameters.CHEMO_MAX / (
                1 + math.exp(parameters.CHEMO_STEEP * (
                    parameters.CHEMO_HALF - 2 * parameters.DX * mag_Gradient_CXCL12)))) * Gradient_CXCL12

            cell.Polarity += chemoFactor

        # Influence due to CXCL13
        if cell.responsiveToCXCL13:
            x_diff = parameters.grid_cxcl13[(x + 1, y, z)] - parameters.grid_cxcl13[(x - 1, y, z)]
            y_diff = parameters.grid_cxcl13[(x, y + 1, z)] - parameters.grid_cxcl13[(x, y - 1, z)]
            z_diff = parameters.grid_cxcl13[(x, y, z + 1)] - parameters.grid_cxcl13[(x, y, z - 1)]

            Gradient_CXCL13 = np.array(
                [x_diff / (2 * parameters.DX), y_diff / (2 * parameters.DX), z_diff / (2 * parameters.DX)])
            mag_Gradient_CXCL13 = np.linalg.norm(Gradient_CXCL13)
            chemoFactor = (parameters.CHEMO_MAX / (
                1 + math.exp(parameters.CHEMO_STEEP * (
                    parameters.CHEMO_HALF - 2 * parameters.DX * mag_Gradient_CXCL13)))) * Gradient_CXCL13

            cell.Polarity += chemoFactor

        # Influence specific to T Cells
        if cell_type == CellType.TCell:
            cell.Polarity = ((1.0 - parameters.NORTH_WEIGHT) * cell.Polarity) + (
                parameters.NORTH * parameters.NORTH_WEIGHT)

        cell.Polarity = cell.Polarity / np.linalg.norm(cell.Polarity)

    # Probability of movement
    pDifu = speed * parameters.DT / parameters.DX

    if random.uniform(0, 1) < pDifu:
        # Find possible new positions based in order of best preference
        WantedPosition = np.asarray(pos) + cell.Polarity
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
                cell.Position = new_pos

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
    cell = parameters.cell_properties[id]

    v = np.random.standard_normal(3)
    cell.Polarity = v / np.linalg.norm(v)


def initiate_cycle(id, parameters):
    """
    Algorithm 3, Updating events at the Centroblast Stage.
    Sets the state for a cell depending on how many divisions left to complete.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """

    cell = parameters.cell_properties[id]
    # noinspection PyUnresolvedReferences
    if cell.numDivisionsToDo == 0:
        cell.State = CellState.Stop_Dividing
    else:
        cell.State = CellState.cb_G1
        cell.cycleStartTime = 0
        cell.endOfThisPhase = get_duration(cell.State)


def progress_cycle(id, parameters):
    """
    Algorithm 4, Updating events at the Centroblast Stage.
    Progresses the cell to its next state and calculates how long it till next change of state.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    cell = parameters.cell_properties[id]
    # Progress cell into its next state
    cell.cycleStartTime += parameters.DT
    if cell.cycleStartTime > cell.endOfThisPhase:
        if cell.State == CellState.cb_G1:
            cell.State = CellState.cb_S
        elif cell.State == CellState.cb_S:
            cell.State = CellState.cb_G2
        elif cell.State == CellState.cb_G2:
            cell.State = CellState.cb_divide

        # Finds time till end of new state.
        if cell.State not in [CellState.cb_divide, CellState.Stop_Dividing]:
            cell.endOfThisPhase = get_duration(cell.State)
            cell.cycleStartTime = 0


def divide_and_mutate(id, parameters):
    # TODO make sure cell doesnt divide into wrong zone
    """
    Algorithm 4, Updating events at the Centroblast Stage.
    Takes a cell and divides it with probability pNow. Will also attempt to mutate the cells.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    old_cell = parameters.cell_properties[id]
    if random.uniform(0, 1) < parameters.PROB_NOW:
        # Find all empty positions neighbouring cell.
        pos = old_cell.Position
        empty_neighbours = [tuple(np.array(pos) + np.array(possible_neighbour)) for possible_neighbour in
                            parameters.POSSIBLE_NEIGHBOURS if np.linalg.norm(
                np.asarray(possible_neighbour) + np.asarray(pos) - np.array(parameters.OFFSET)) <= (parameters.N / 2)]

        # Randomly choose one position for new cell
        if empty_neighbours:
            divide_pos = random.choice(empty_neighbours)

            # Generate a new ID for the cell and copy over the properties from the old cell.
            new_id = parameters.available_cell_ids.pop()
            parameters.list_cb.append(new_id)

            new_cell = copy.copy(old_cell)
            new_cell.Polarity = np.copy(old_cell.Polarity)
            new_cell.Position = divide_pos
            new_cell.retainedAg = None
            parameters.cell_properties[new_id] = new_cell

            parameters.Grid_ID[divide_pos] = new_id
            parameters.Grid_Type[divide_pos] = new_cell.Type

            old_cell.numDivisionsToDo -= 1
            new_cell.numDivisionsToDo -= 1

            old_cell.IAmHighAg = False
            new_cell.IAmHighAg = False

            # Initiate the cycle for each cell.
            initiate_cycle(id, parameters)
            initiate_cycle(new_id, parameters)

            # Mutate the cells
            if parameters.t > parameters.MUTATION_START_TIME:
                mutate(id, parameters.cell_properties)
                mutate(new_id, parameters.cell_properties)

            # Assign amount of retained antigen to each cell
            if random.uniform(0, 1) < parameters.PROB_DIVIDE_AG_ASYMMETRIC:
                if old_cell.retainedAg == 0:
                    new_cell.retainedAg = 0
                else:
                    sep = random.gauss(parameters.POLARITY_INDEX, 1)
                    while sep < 0 or sep > 1:
                        sep = random.gauss(parameters.POLARITY_INDEX, 1)

                    new_cell.retainedAg = sep * old_cell.retainedAg
                    old_cell.retainedAg = (1 - sep) * old_cell.retainedAg

                    if sep > 0.5:
                        new_cell.IAmHighAg = True
                    else:
                        old_cell.IAmHighAg = False
            else:
                new_cell.retainedAg = old_cell.retainedAg / 2
                old_cell.retainedAg = old_cell.retainedAg / 2


def progress_fdc_selection(id, parameters):
    """
    Algorithm 5, Antigen Collection from FDCs.
    Allows for centrocytes to collect antigen from neighbouring F cells / fragments.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    cell = parameters.cell_properties[id]
    if cell.State == CellState.Unselected:
        # Progress selected clock and check if able to collect antigen.
        cell.selectedClock += parameters.DT
        if cell.selectedClock <= parameters.COLLECT_FDC_PERIOD:
            cell.Clock += parameters.DT
            # noinspection PyTypeChecker
            if cell.Clock > parameters.TEST_DELAY:
                cell.selectable = True
                # Find frag component with largest amount of antigen.
                frag_max = None
                frag_max_id = None
                pos = cell.Position
                for neighbour in parameters.POSSIBLE_NEIGHBOURS:
                    neighbour_pos = tuple(np.array(pos) + np.array(neighbour))
                    if parameters.Grid_Type[neighbour_pos] in [CellType.Fragment, CellType.FCell]:
                        frag_id = parameters.Grid_ID[neighbour_pos]
                        frag_cell = parameters.cell_properties[frag_id]
                        if frag_cell.antigenAmount > frag_max:
                            frag_max = frag_cell.antigenAmount
                            frag_max_id = frag_id

                pBind = affinity(cell.BCR) * frag_max / parameters.ANTIGEN_SATURATION
                # Bind cell and fragment with probability pBind or reset clock.
                if random.choice(0, 1) < pBind:
                    cell.State = CellState.FDCcontact
                    cell.FragContact = frag_max_id
                else:
                    cell.Clock = 0
                    cell.selectable = False
        else:
            # Cell dies if it doesn't get any contacts.
            if cell.numFDCContacts == 0:
                cell.State = CellState.Apoptosis
            else:
                cell.State = CellState.FDCselected

    elif cell.State == CellState.Contact:
        cell.selectedClock += parameters.DT
        if random.uniform(0, 1) < parameters.P_SEL:
            cell.numFDCContacts += 1
            frag_id = cell.FragContact
            frag_cell = parameters.cell_properties[frag_id]
            frag_cell.antigenAmount -= 1
            cell.State = CellState.Unselected
            cell.Clock = 0
            cell.selectable = False


# noinspection PyTypeChecker,PyTypeChecker,PyTypeChecker
def progress_tcell_selection(id, parameters):
    """
    Algorithm 6, Screening for T cell help at Centrocyte Stage.

    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    cell = parameters.cell_properties[id]
    if cell.State == CellState.FDCselected:
        # Find if there is a neighbouring T cell.
        pos = cell.Position
        for neighbour in parameters.POSSIBLE_NEIGHBOURS:
            neighbour_pos = tuple(np.array(pos) + np.array(neighbour))
            # If there is a neighbouring T cell, we record the contact.
            if parameters.Grid_Type[neighbour_pos] == CellType.TCell and cell.State != CellState.TCcontact:
                update_tcell(id, parameters.Grid_ID[neighbour_pos], parameters)
                cell.State = CellState.TCcontact
                cell.tcClock = 0
                cell.tcSignalDuration = 0

    elif cell.State == CellState.TCcontact:
        cell.tcClock += parameters.DT
        tcell = parameters.cell_properties[cell.TCell_Contact]
        # Check is current cell has least amount of antigens compared to T cells neighbouring cells.
        lowest_antigen = True
        for ID_BC in tcell.BCell_Contacts:
            other_cell = parameters.cell_properties[ID_BC]
            if id != ID_BC and cell.retainedAg <= other_cell.retainedAg:
                lowest_antigen = False
        if lowest_antigen:
            cell.tcSignalDuration += parameters.DT
        if cell.tcSignalDuration > parameters.TC_RESCUE_TIME:
            cell.State = CellState.Selected
            cell.selectedClock = 0
            rand = random.uniform(0, 1)
            cell.IndividualDifDelay = parameters.DIF_DELAY * (1 + 0.1 * math.log(1 - rand) / rand)
            liberate_tcell(id, cell.TCell_Contact, parameters.cell_properties)
        elif cell.tcClock >= parameters.TC_TIME:
            cell.State = CellState.Apoptosis
            liberate_tcell(id, cell.TCell_Contact, parameters.cell_properties)

    elif cell.State == CellState.Selected:
        cell.selectedClock += parameters.DT
        if cell.selectedClock > cell.IndividualDifDelay:
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
    TCell = parameters.cell_properties[id_t]
    BCell = parameters.cell_properties[id_b]
    TCell.BCell_Contacts.append(id_b)
    BCell.TCell_Contact = id_t
    TCell.State = CellState.TC_CC_Contact


def liberate_tcell(id_b, id_t, parameters):
    """
    Algorithm 7, Updating the T cells according to B cells interactions.
    Updates records of whether cells are interacting, specifically removing cells from being interactive.
    :param id_b: integer, determines which B cell in the pop we are talking about.
    :param id_t: integer, determines which T cell in the pop we are talking about.
    :return:
    """
    TCell = parameters.cell_properties[id_t]
    BCell = parameters.cell_properties[id_b]
    TCell.BCell_Contacts.remove(id_b)
    BCell.TCell_Contact = None
    if not TCell.BCell_Contacts:
        TCell.State = CellState.TCnormal


def differ_to_out(id, parameters):
    """
    Algorithm 8, Transition between Centroblasts, Centrocyctes, and Output Cells.
    Transition a cell from Centroblast or Centrocycte to an Output Cell.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    parameters.list_outcells.append(id)
    old_cell = parameters.cell_properties(id)
    parameters.NumBCROutCells[old_cell.BCR] += 1
    # Update cell and Grid position properties.
    new_cell = SimpleNamespace(Type=CellType.OutCell, Position=old_cell.Position, Polarity=old_cell.Polarity,
                               responsiveToCXCL12=None, responsiveToCXCL13=None)
    parameters.cell_properties[id] = new_cell
    parameters.Grid_Type[new_cell.Position] = CellType.OutCell
    initiate_chemokine_receptors(id, parameters)


def differ_to_cb(id, parameters):
    """
    Algorithm 8, Transition between Centroblasts, Centrocyctes, and Output Cells.
    Transitions a cell from Centrocyte to Centroblast.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    parameters.list_cb.append(id)
    old_cell = parameters.cell_properties[id]

    # Update grid and cell details.
    parameters.Grid_Type[old_cell.Position] = CellType.Centroblast
    new_cell = SimpleNamespace(Type=CellType.Centroblast, Position=old_cell.Position, State=None, BCR=old_cell.BCR,
                               Polarity=old_cell.Polarity, responsiveToCXCL12=None,
                               responsiveToCXCL13=None, numDivisionsToDo=None, pMutation=None, IAmHighAg=True,
                               retainedAg=old_cell.NumFDCContacts, cycleStartTime=None, endOfThisPhase=None)

    parameters.cell_properties[id] = new_cell
    # Find number of divisions to do.
    agFactor = old_cell.numFDCContacts ** parameters.pMHCdepHill
    new_cell.numDivisionsToDo = parameters.pMHCdepMin + (parameters.pMHCdepMax - parameters.pMHCdepMin) * agFactor / (
        agFactor + parameters.pMHCdepK ** parameters.pMHCdepHill)

    # Find new probability of mutation.
    new_cell.pMutation = p_mut(parameters.t) + (parameters.PROB_MUT_AFTER_SELECTION - p_mut(parameters.t)) * affinity(
        new_cell.BCR) ** parameters.PROB_MUT_AFFINITY_EXPONENT

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
    old_cell = parameters.cell_properties[id]

    new_cell = SimpleNamespace(Type=CellType.Centrocyte, Position=old_cell.Position, State=CellState.Unselected,
                               BCR=old_cell.BCR, Polarity=old_cell.Polarity, responsiveToCXCL12=None,
                               responsiveToCXCL13=None, selectedClock=0.0, Clock=0.0, selectable=False,
                               FragContact=None,
                               numFDCContacts=None, tcClock=None, tcSignalDuration=None, IndividualDifDelay=None,
                               TCell_Contact=None)
    parameters.cell_properties[id] = new_cell
    initiate_chemokine_receptors(id, parameters.cell_properties)
    parameters.Grid_Type[new_cell.Position] = CellType.Centrocyte
    if parameters.DELETE_AG_IN_FRESH_CC:
        new_cell.numFDCContacts = 0.0
    else:
        new_cell.numFDCContacts = math.floor(old_cell.retainedAg + 0.5)


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
        cell = SimpleNamespace(Type=CellType.Stromal, Position=pos)
        parameters.cell_properties[id] = cell
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
        cell = SimpleNamespace(Type=CellType.FCell, Position=pos, antigenAmount=None, icAmount=0, Fragments=[])
        parameters.cell_properties[id] = cell
        parameters.Grid_ID[pos] = id
        parameters.Grid_Type[pos] = CellType.FCell

        # Find the fragments for the FCell
        fcell_id = id
        fragments = cell.Fragments
        x, y, z = pos
        for i in range(1, parameters.DENDRITE_LENGTH + 1):
            for frag_pos in [(x + i, y, z), (x - i, y, z), (x, y + i, z), (x, y - i, z), (x, y, z - i)]:
                if (frag_pos[0] - (parameters.N / 2 + 0.5)) ** 2 + (frag_pos[1] - (parameters.N / 2 + 0.5)) ** 2 + (
                            frag_pos[2] - (parameters.N / 2 + 0.5)) ** 2 <= (parameters.N / 2) ** 2 and \
                                parameters.Grid_ID[frag_pos] is None:
                    id = parameters.available_cell_ids.pop()
                    fragments.append(id)
                    cell = SimpleNamespace(Type=CellType.Fragment, Position=frag_pos, antigenAmount=None, icAmount=0,
                                           Parent=fcell_id)
                    parameters.cell_properties[id] = cell
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
                cell = SimpleNamespace(Type=CellType.Fragment, Position=frag_pos, antigenAmount=None, icAmount=0.0,
                                       Parent=fcell_id)
                parameters.cell_properties[id] = cell
                parameters.Grid_ID[frag_pos] = id
                parameters.Grid_Type[frag_pos] = CellType.Fragment

        # Assign each fragment an amount of antigen
        fcell_vol = len(fragments) + 1  # +1 accounts for centre
        ag_per_frag = parameters.INITIAL_ANTIGEN_AMOUNT_PER_FDC / fcell_vol
        for id in [fcell_id] + fragments:
            cell = parameters.cell_properties[id]
            cell.antigenAmount = ag_per_frag

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
        cell = SimpleNamespace(Type=CellType.Centroblast, Position=pos, State=None,
                               BCR=random.choice(tuple(parameters.BCR_values_all)), Polarity=polarity_vector,
                               responsiveToCXCL12=None,
                               responsiveToCXCL13=None, numDivisionsToDo=parameters.NUM_DIV_INITIAL_CELLS,
                               pMutation=p_mut(0.0),
                               IAmHighAg=False, retainedAg=0.0,
                               cycleStartTime=None, endOfThisPhase=None)
        parameters.cell_properties[id] = cell

        initiate_cycle(id, parameters)
        initiate_chemokine_receptors(id, parameters)

    # Initialise T Cells:
    for i in range(parameters.INITIAL_NUM_TC):
        pos = random.choice(parameters.LIGHT_ZONE)  # Find empty location in light zone
        while parameters.Grid_ID[pos] is not None:
            pos = random.choice(parameters.LIGHT_ZONE)

        # Obtain new ID for new cell.
        id = parameters.available_cell_ids.pop()

        # Add cell to appropriate lists and dictionaries
        parameters.list_tc.append(id)
        polarity_vector = np.random.standard_normal(3)
        polarity_vector = polarity_vector / np.linalg.norm(polarity_vector)
        cell = SimpleNamespace(Type=CellType.TCell, Position=pos, State=CellState.TCnormal, Polarity=polarity_vector,
                               responsiveToCXCL12=False, responsiveToCXCL13=False, BCell_Contacts=[])
        parameters.cell_properties[id] = cell
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
            cell = parameters.cell_properties[id]
            # Secrete CXCL13 from Fcells
            signal_secretion(id, 'CXCL13', parameters.p_mkCXCL13)
            fragments = cell.Fragments
            # Update antigen amounts for each fragment.
            for frag_id in fragments:
                frag = parameters.cell_properties[frag_id]
                for bcr_seq in parameters.BCR_values_all:
                    d_ic = parameters.DT * (
                        parameters.K_ON * frag.antigenAmount * parameters.AntibodyPerBCR[bcr_seq] - k_off(
                            bcr_seq) * frag.icAmount)
                    frag.antigenAmount -= d_ic
                    frag.icAmount += d_ic

        # Update the number of outcells and amount of antibody for each CR value.
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
            cell = parameters.cell_properties[id]
            # Move cell and remove if on surface of sphere / Germinal Center.
            move(id, parameters)
            pos = cell.Position
            if is_surface_point(pos):
                parameters.list_outcells.remove(id)
                parameters.available_cell_ids.append(id)

        # Randomly iterate over Centroblast cells.
        random.shuffle(parameters.list_cb)
        for id in parameters.list_cb:
            cell = parameters.cell_properties[id]
            # Progress cells in their lifetime cycle.
            update_chemokines_receptors(id, parameters)
            progress_cycle(id, parameters)
            # Attempt to divide cell if ready.
            if cell.State == CellState.cb_divide:
                divide_and_mutate(id, parameters)
            if cell.State == CellState.Stop_Dividing:
                if random.uniform(0, 1) < parameters.PROB_DIF:
                    if cell.IAmHighAg:
                        differ_to_out(id, parameters)
                        parameters.list_cb.remove(id)
                    else:
                        differ_to_cc(id, parameters)
                        parameters.list_cb.remove(id)

            # Move allowed cells.
            if cell.State != CellState.cb_M:
                move(id, parameters)

        # Randomly iterate over Centrocyte cells.
        random.shuffle(parameters.list_cc)
        for id in parameters.list_cc:
            cell = parameters.cell_properties[id]
            # Update cells progress
            update_chemokines_receptors(id, parameters)
            progress_fdc_selection(id, parameters)
            progress_tcell_selection(id, parameters)
            # Remove cell from simulation if dead or move if not in contact with T or F cell.
            if cell.State == CellState.Apoptosis:
                parameters.list_cb.remove(id)
                parameters.available_cell_ids.append(id)
            elif cell.State not in [CellState.FDCcontact, CellState.TCcontact]:
                move(id, parameters)

        # Randomly iterate over T cells and move if not attached to another cell.
        random.shuffle(parameters.list_tc)
        for id in parameters.list_tc:
            cell = parameters.cell_properties[id]
            if cell.State == CellState.TCnormal:
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

        # Numpy Array to store each cell's attributes
        self.cell_properties = [None] * len(self.ALL_POINTS)

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

        # Dictionaries storing what is at each location. Initially empty, so 'None'.
        self.Grid_ID = {pos: None for pos in self.ALL_POINTS}
        self.Grid_Type = {pos: None for pos in self.ALL_POINTS}

        # Dictionaries storing amounts of CXCL12 and CXCL13 at each point:

        self.grid_cxcl12 = {pos: random.uniform(80e-11, 80e-10) for pos in
                            [(x + self.N / 2 + 1, y + self.N / 2 + 1, z + self.N / 2 + 1) for x in
                             range(-self.N // 2 - 1, self.N // 2 + 1) for y in
                             range(-self.N // 2 - 1, self.N // 2 + 1)
                             for z in range(-self.N // 2 - 1, self.N // 2 + 1)]}
        self.grid_cxcl13 = {pos: random.uniform(0.1e-10, 0.1e-9) for pos in
                            [(x + self.N / 2 + 1, y + self.N / 2 + 1, z + self.N / 2 + 1) for x in
                             range(-self.N // 2 - 1, self.N // 2 + 1) for y in
                             range(-self.N // 2 - 1, self.N // 2 + 1)
                             for z in range(-self.N // 2 - 1, self.N // 2 + 1)]}

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
