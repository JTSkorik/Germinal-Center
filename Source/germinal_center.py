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

def mutate(id, cell_properties):
    """
    Algorithm 1, Mutation.
    Mutates the BCR value for a cell with a certain probability.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    cell = cell_properties[id]
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
    if cell.BCR not in BCR_values_all:
        BCR_values_all.add(cell.BCR)
        NumBCROutCells[cell.BCR] = 0
        NumBCROutCellsProduce[cell.BCR] = 0
        AntibodyPerBCR[cell.BCR] = 0

    return cell_properties


def initiate_chemokine_receptors(id, cell_properties):
    """
    Algorithm 2, Dynamic Updating of Chemotaxis.
    Used to determine the attributes responsive_to_CXCL12 and responsive_to_CXCL13 of a cell given its type.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    cell = cell_properties[id]
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

    return cell_properties


def update_chemokines_receptors(id, cell_properties, grid_cxcl12, grid_cxcl13):
    """
    Algorithm 2, Dynamic Updating of Chemotaxis.
    Used to update the attributes responsive_to_CXCL12 and responsive_to_CXCL13 of a cell given its type and state.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    cell = cell_properties[id]
    pos = cell.Position
    if cell.Type == CellType.Centrocyte:
        if cell.State == CellState.Unselected:
            if grid_cxcl13[pos] > CXCL13_CRIT:
                cell.responsiveToCXCL13 = False
            elif grid_cxcl13[pos] < CXCL13_RECRIT:
                cell.responsiveToCXCL13 = True
    elif cell.Type == CellType.Centroblast:
        if grid_cxcl12[pos] > CXCL12_CRIT:
            cell.responsiveToCXCL12 = False
        elif grid_cxcl12[pos] < CXCL12_RECRIT:
            cell.responsiveToCXCL12 = True

    return cell_properties


def move(id, cell_properties, grid_cxcl12, grid_cxcl13):
    # TODO add check that the cells are staying inside their respective zones.
    """
    Algorithm 3, Updating Position and Polarity of cells at each time-point.
    Updates the polarity of a cell and then will move the cell within the Germinal center. Both events occur stochastically.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    cell = cell_properties[id]
    # Obtain current position of cell.
    pos = cell.Position
    x, y, z = pos

    # Obtain required parameters
    cell_type = cell.Type
    if cell_type == CellType.Centrocyte:
        prob = PLT_CENTROCYTE
        speed = SPEED_CENTROCYTE
    elif cell_type == CellType.Centroblast:
        prob = PLT_CENTROBLAST
        speed = SPEED_CENTROBLAST
    elif cell_type == CellType.TCell:
        prob = PLT_TCELL
        speed = SPEED_TCELL
    elif cell_type == CellType.OutCell:
        prob = PLT_OUTCELL
        speed = SPEED_OUTCELL
    else:
        prob = None
        speed = None
        print("move: Invalid cell_type, {}".format(cell_type))

    # Calculate new polarity
    if random.uniform(0, 1) < prob:
        # Turning angles influence
        theta = random.gauss(0, 1)
        phi = random.uniform(0, 2 * math.pi)
        cell_properties = turn_angle(id, cell_properties, theta, phi)

        # Influence due to CXCL12
        if cell.responsiveToCXCL12:
            x_diff = grid_cxcl12[(x + 1, y, z)] - grid_cxcl12[(x - 1, y, z)]
            y_diff = grid_cxcl12[(x, y + 1, z)] - grid_cxcl12[(x, y - 1, z)]
            z_diff = grid_cxcl12[(x, y, z + 1)] - grid_cxcl12[(x, y, z - 1)]

            Gradient_CXCL12 = np.array([x_diff / (2 * DX), y_diff / (2 * DX), z_diff / (2 * DX)])

            mag_Gradient_CXCL12 = np.linalg.norm(Gradient_CXCL12)
            chemoFactor = (CHEMO_MAX / (
                1 + math.exp(CHEMO_STEEP * (CHEMO_HALF - 2 * DX * mag_Gradient_CXCL12)))) * Gradient_CXCL12

            cell.Polarity += chemoFactor

        # Influence due to CXCL13
        if cell.responsiveToCXCL13:
            x_diff = grid_cxcl13[(x + 1, y, z)] - grid_cxcl13[(x - 1, y, z)]
            y_diff = grid_cxcl13[(x, y + 1, z)] - grid_cxcl13[(x, y - 1, z)]
            z_diff = grid_cxcl13[(x, y, z + 1)] - grid_cxcl13[(x, y, z - 1)]

            Gradient_CXCL13 = np.array([x_diff / (2 * DX), y_diff / (2 * DX), z_diff / (2 * DX)])
            mag_Gradient_CXCL13 = np.linalg.norm(Gradient_CXCL13)
            chemoFactor = (CHEMO_MAX / (
                1 + math.exp(CHEMO_STEEP * (CHEMO_HALF - 2 * DX * mag_Gradient_CXCL13)))) * Gradient_CXCL13

            cell.Polarity += chemoFactor

        # Influence specific to T Cells
        if cell_type == CellType.TCell:
            cell.Polarity = ((1.0 - NORTH_WEIGHT) * cell.Polarity) + (NORTH * NORTH_WEIGHT)

        cell.Polarity = cell.Polarity / np.linalg.norm(cell.Polarity)

    # Probability of movement
    pDifu = speed * DT / DX

    if random.uniform(0, 1) < pDifu:
        # Find possible new positions based in order of best preference
        WantedPosition = np.asarray(pos) + cell.Polarity
        Neighbours = [np.asarray(Movement) + np.asarray(pos) for Movement in
                      POSSIBLE_NEIGHBOURS if np.linalg.norm(
                np.asarray(Movement) + np.asarray(pos) - np.array(OFFSET)) <= (N / 2)]
        Neighbours.sort(key=lambda w: np.linalg.norm(w - WantedPosition))

        # Move the cell to best available position that isn't against direction of polarity
        count = 0
        moved = False
        while not moved and count <= 9:
            new_pos = tuple(Neighbours[count])
            if Grid_ID[new_pos] is None:
                cell.Position = new_pos

                Grid_ID[new_pos] = Grid_ID[pos]
                Grid_Type[new_pos] = Grid_Type[pos]
                Grid_ID[pos] = None
                Grid_Type[pos] = None

                moved = True
            count += 1

    return cell_properties


def turn_angle(id, cell_properties, theta, phi):
    """
    Algorithm 3 (Updating Position and Polarity of cells at each time-point)
    Rotates the polarity of a cell by the given angles.
    Yet to be finished.
    :param id: integer, determines which cell in the pop we are talking about.
    :param theta: float, randomly generated turning angle.
    :param phi: float, another randomly generated turning angle between 0 and 2pi.
    :return:
    """
    cell = cell_properties[id]

    v = np.random.standard_normal(3)
    cell.Polarity = v / np.linalg.norm(v)

    return cell_properties


def initiate_cycle(id, cell_properties):
    """
    Algorithm 3, Updating events at the Centroblast Stage.
    Sets the state for a cell depending on how many divisions left to complete.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """

    cell = cell_properties[id]
    # noinspection PyUnresolvedReferences
    if cell.numDivisionsToDo == 0:
        cell.State = CellState.Stop_Dividing
    else:
        cell.State = CellState.cb_G1
        cell.cycleStartTime = 0
        cell.endOfThisPhase = get_duration(cell.State)

    return cell_properties


def progress_cycle(id, cell_properties):
    """
    Algorithm 4, Updating events at the Centroblast Stage.
    Progresses the cell to its next state and calculates how long it till next change of state.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    cell = cell_properties[id]
    # Progress cell into its next state
    cell.cycleStartTime += DT
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

    return cell_properties


def divide_and_mutate(id, cell_properties, available_cell_ids, t, list_cb):
    # TODO make sure cell doesnt divide into wrong zone
    """
    Algorithm 4, Updating events at the Centroblast Stage.
    Takes a cell and divides it with probability pNow. Will also attempt to mutate the cells.
    :param id: integer, determines which cell in the pop we are talking about.
    :param t: float, current time of the simulation.
    :return:
    """
    old_cell = cell_properties[id]
    if random.uniform(0, 1) < PROB_NOW:
        # Find all empty positions neighbouring cell.
        pos = old_cell.Position
        empty_neighbours = [tuple(np.array(pos) + np.array(possible_neighbour)) for possible_neighbour in
                            POSSIBLE_NEIGHBOURS if np.linalg.norm(
                np.asarray(possible_neighbour) + np.asarray(pos) - np.array(OFFSET)) <= (N / 2)]

        # Randomly choose one position for new cell
        if empty_neighbours:
            divide_pos = random.choice(empty_neighbours)

            # Generate a new ID for the cell and copy over the properties from the old cell.
            new_id = available_cell_ids.pop()
            list_cb.append(new_id)

            new_cell = copy.copy(old_cell)
            new_cell.Polarity = np.copy(old_cell.Polarity)
            new_cell.Position = divide_pos
            new_cell.retainedAg = None
            cell_properties[new_id] = new_cell

            Grid_ID[divide_pos] = new_id
            Grid_Type[divide_pos] = new_cell.Type

            old_cell.numDivisionsToDo -= 1
            new_cell.numDivisionsToDo -= 1

            old_cell.IAmHighAg = False
            new_cell.IAmHighAg = False

            # Initiate the cycle for each cell.
            cell_properties = initiate_cycle(id, cell_properties)
            cell_properties = initiate_cycle(new_id, cell_properties)

            # Mutate the cells
            if t > MUTATION_START_TIME:
                cell_properties = mutate(id, cell_properties)
                cell_properties = mutate(new_id, cell_properties)

            # Assign amount of retained antigen to each cell
            if random.uniform(0, 1) < PROB_DIVIDE_AG_ASYMMETRIC:
                if old_cell.retainedAg == 0:
                    new_cell.retainedAg = 0
                else:
                    sep = random.gauss(POLARITY_INDEX, 1)
                    while sep < 0 or sep > 1:
                        sep = random.gauss(POLARITY_INDEX, 1)

                    new_cell.retainedAg = sep * old_cell.retainedAg
                    old_cell.retainedAg = (1 - sep) * old_cell.retainedAg

                    if sep > 0.5:
                        new_cell.IAmHighAg = True
                    else:
                        old_cell.IAmHighAg = False
            else:
                new_cell.retainedAg = old_cell.retainedAg / 2
                old_cell.retainedAg = old_cell.retainedAg / 2

    return cell_properties, available_cell_ids, list_cb


def progress_fdc_selection(id, cell_properties):
    """
    Algorithm 5, Antigen Collection from FDCs.
    Allows for centrocytes to collect antigen from neighbouring F cells / fragments.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    cell = cell_properties[id]
    if cell.State == CellState.Unselected:
        # Progress selected clock and check if able to collect antigen.
        cell.selectedClock += DT
        if cell.selectedClock <= COLLECT_FDC_PERIOD:
            cell.Clock += DT
            # noinspection PyTypeChecker
            if cell.Clock > TEST_DELAY:
                cell.selectable = True
                # Find frag component with largest amount of antigen.
                frag_max = None
                frag_max_id = None
                pos = cell.Position
                for neighbour in POSSIBLE_NEIGHBOURS:
                    neighbour_pos = tuple(np.array(pos) + np.array(neighbour))
                    if Grid_Type[neighbour_pos] in [CellType.Fragment, CellType.FCell]:
                        frag_id = Grid_ID[neighbour_pos]
                        frag_cell = cell_properties[frag_id]
                        if frag_cell.antigenAmount > frag_max:
                            frag_max = frag_cell.antigenAmount
                            frag_max_id = frag_id

                pBind = affinity(cell.BCR) * frag_max / ANTIGEN_SATURATION
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
        cell.selectedClock += DT
        if random.uniform(0, 1) < P_SEL:
            cell.numFDCContacts += 1
            frag_id = cell.FragContact
            frag_cell = cell_properties[frag_id]
            frag_cell.antigenAmount -= 1
            cell.State = CellState.Unselected
            cell.Clock = 0
            cell.selectable = False

    return cell_properties


# noinspection PyTypeChecker,PyTypeChecker,PyTypeChecker
def progress_tcell_selection(id, cell_properties, t, list_cc, list_outcells, list_cb):
    """
    Algorithm 6, Screening for T cell help at Centrocyte Stage.

    :param id: integer, determines which cell in the pop we are talking about.
    :param t: t: float, current time of the simulation.
    :return:
    """
    cell = cell_properties[id]
    if cell.State == CellState.FDCselected:
        # Find if there is a neighbouring T cell.
        pos = cell.Position
        for neighbour in POSSIBLE_NEIGHBOURS:
            neighbour_pos = tuple(np.array(pos) + np.array(neighbour))
            # If there is a neighbouring T cell, we record the contact.
            if Grid_Type[neighbour_pos] == CellType.TCell and cell.State != CellState.TCcontact:
                cell_properties = update_tcell(id, Grid_ID[neighbour_pos], cell_properties)
                cell.State = CellState.TCcontact
                cell.tcClock = 0
                cell.tcSignalDuration = 0

    elif cell.State == CellState.TCcontact:
        cell.tcClock += DT
        tcell = cell_properties[cell.TCell_Contact]
        # Check is current cell has least amount of antigens compared to T cells neighbouring cells.
        lowest_antigen = True
        for ID_BC in tcell.BCell_Contacts:
            other_cell = cell_properties[ID_BC]
            if id != ID_BC and cell.retainedAg <= other_cell.retainedAg:
                lowest_antigen = False
        if lowest_antigen:
            cell.tcSignalDuration += DT
        if cell.tcSignalDuration > TC_RESCUE_TIME:
            cell.State = CellState.Selected
            cell.selectedClock = 0
            rand = random.uniform(0, 1)
            cell.IndividualDifDelay = DIF_DELAY * (1 + 0.1 * math.log(1 - rand) / rand)
            cell_properties = liberate_tcell(id, cell.TCell_Contact, cell_properties)
        elif cell.tcClock >= TC_TIME:
            cell.State = CellState.Apoptosis
            cell_properties = liberate_tcell(id, cell.TCell_Contact, cell_properties)

    elif cell.State == CellState.Selected:
        cell.selectedClock += DT
        if cell.selectedClock > cell.IndividualDifDelay:
            if random.uniform(0, 1) < PROB_DIF:
                if random.uniform(0, 1) < PROB_DIF_TO_OUT:
                    cell_properties, list_outcells, list_cc = differ_to_out(id, cell_properties, list_outcells, list_cc)
                else:
                    cell_properties, list_cb, list_cc = differ_to_cb(id, cell_properties, t, list_cb, list_cc)

    return cell_properties, list_cc, list_outcells, list_cb


def update_tcell(id_b, id_t, cell_properties):
    """
    Algorithm 7, Updating the T cells according to B cells interactions.
    Updates records of whether cells are interacting, specifically marking two cells as interacting.
    :param id_b: integer, determines which B cell in the pop we are talking about.
    :param id_t: integer, determines which T cell in the pop we are talking about.
    :return:
    """
    TCell = cell_properties[id_t]
    BCell = cell_properties[id_b]
    TCell.BCell_Contacts.append(id_b)
    BCell.TCell_Contact = id_t
    TCell.State = CellState.TC_CC_Contact

    return cell_properties


def liberate_tcell(id_b, id_t, cell_properties):
    """
    Algorithm 7, Updating the T cells according to B cells interactions.
    Updates records of whether cells are interacting, specifically removing cells from being interactive.
    :param id_b: integer, determines which B cell in the pop we are talking about.
    :param id_t: integer, determines which T cell in the pop we are talking about.
    :return:
    """
    TCell = cell_properties[id_t]
    BCell = cell_properties[id_b]
    TCell.BCell_Contacts.remove(id_b)
    BCell.TCell_Contact = None
    if not TCell.BCell_Contacts:
        TCell.State = CellState.TCnormal

    return cell_properties


def differ_to_out(id, cell_properties, list_outcells, cell_state_list):
    """
    Algorithm 8, Transition between Centroblasts, Centrocyctes, and Output Cells.
    Transition a cell from Centroblast or Centrocycte to an Output Cell.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    list_outcells.append(id)
    cell_state_list.remove(id)
    old_cell = cell_properties(id)
    NumBCROutCells[old_cell.BCR] += 1
    # Update cell and Grid position properties.
    new_cell = SimpleNamespace(Type=CellType.OutCell, Position=old_cell.Position, Polarity=old_cell.Polarity,
                               responsiveToCXCL12=None, responsiveToCXCL13=None)
    cell_properties[id] = new_cell
    Grid_Type[new_cell.Position] = CellType.OutCell
    cell_properties = initiate_chemokine_receptors(id, cell_properties)

    return cell_properties, list_outcells, cell_state_list


def differ_to_cb(id, cell_properties, t, list_cb, cell_state_list):
    """
    Algorithm 8, Transition between Centroblasts, Centrocyctes, and Output Cells.
    Transitions a cell from Centrocyte to Centroblast.
    :param id: integer, determines which cell in the pop we are talking about.
    :param t: float,  current time of the simulation.
    :return:
    """
    list_cb.append(id)
    cell_state_list.remove(id)
    old_cell = cell_properties[id]

    # Update grid and cell details.
    Grid_Type[old_cell.Position] = CellType.Centroblast
    new_cell = SimpleNamespace(Type=CellType.Centroblast, Position=old_cell.Position, State=None, BCR=old_cell.BCR,
                               Polarity=old_cell.Polarity, responsiveToCXCL12=None,
                               responsiveToCXCL13=None, numDivisionsToDo=None, pMutation=None, IAmHighAg=True,
                               retainedAg=old_cell.NumFDCContacts, cycleStartTime=None, endOfThisPhase=None)

    cell_properties[id] = new_cell
    # Find number of divisions to do.
    agFactor = old_cell.numFDCContacts ** pMHCdepHill
    new_cell.numDivisionsToDo = pMHCdepMin + (pMHCdepMax - pMHCdepMin) * agFactor / (agFactor + pMHCdepK ** pMHCdepHill)

    # Find new probability of mutation.
    new_cell.pMutation = p_mut(t) + (PROB_MUT_AFTER_SELECTION - p_mut(t)) * affinity(
        new_cell.BCR) ** PROB_MUT_AFFINITY_EXPONENT

    # Initiate cell.
    cell_properties = initiate_chemokine_receptors(id, cell_properties)
    cell_properties = initiate_cycle(id, cell_properties)

    return cell_properties, list_cb, cell_state_list


def differ_to_cc(id, cell_properties, list_cc, cell_state_list):
    """
    Algorithm 8, Transition between Centroblasts, Centrocyctes, and Output Cells.
    Transitions a cell from Centroblast to Centrocyte.
    :param id: integer, determines which cell in the pop we are talking about.
    :return:
    """
    list_cc.append(id)
    cell_state_list.remove(id)
    old_cell = cell_properties[id]

    new_cell = SimpleNamespace(Type=CellType.Centrocyte, Position=old_cell.Position, State=CellState.Unselected,
                               BCR=old_cell.BCR, Polarity=old_cell.Polarity, responsiveToCXCL12=None,
                               responsiveToCXCL13=None, selectedClock=0.0, Clock=0.0, selectable=False,
                               FragContact=None,
                               numFDCContacts=None, tcClock=None, tcSignalDuration=None, IndividualDifDelay=None,
                               TCell_Contact=None)
    cell_properties[id] = new_cell
    cell_properties = initiate_chemokine_receptors(id, cell_properties)
    Grid_Type[new_cell.Position] = CellType.Centrocyte
    if DELETE_AG_IN_FRESH_CC:
        new_cell.numFDCContacts = 0.0
    else:
        new_cell.numFDCContacts = math.floor(old_cell.retainedAg + 0.5)

    return cell_properties, list_cc, cell_state_list


def initialise_cells():
    """
    Algorithm 9, Initialisation.
    Starts the simulation by introducing various amounts of the possible cells into the simulation.
    :return:
    """
    # Available Cell IDs:
    available_cell_ids = list(range(len(ALL_POINTS)))

    # Numpy Array to store each cell's attributes
    cell_properties = [None] * len(ALL_POINTS)

    # Lists to store ID of each cell in each state (and fragments)
    list_stromal = []
    list_fdc = []
    list_cb = []
    list_cc = []
    list_tc = []
    list_outcells = []

    # Initialise Stromal Cells:
    for _ in range(INITIAL_NUM_STROMAL_CELLS):
        # Find empty location in dark zone
        pos = random.choice(DARK_ZONE)
        while Grid_ID[pos] is not None:
            pos = random.choice(DARK_ZONE)

        # Obtain new ID for new cell
        id = available_cell_ids.pop()

        # Add to appropriate lists and dictionaries
        list_stromal.append(id)
        cell = SimpleNamespace(Type=CellType.Stromal, Position=pos)
        cell_properties[id] = cell
        Grid_ID[pos] = id
        Grid_Type[pos] = CellType.Stromal

    # Initialise Fragments:
    for _ in range(INITIAL_NUM_FDC):
        # Find empty location in light zone
        pos = random.choice(LIGHT_ZONE)
        while Grid_ID[pos] is not None:
            pos = random.choice(LIGHT_ZONE)

        # Obtain new ID for new cell
        id = available_cell_ids.pop()

        # Add to appropriate lists and dictionaries
        list_fdc.append(id)
        cell = SimpleNamespace(Type=CellType.FCell, Position=pos, antigenAmount=None, icAmount=0, Fragments=[])
        cell_properties[id] = cell
        Grid_ID[pos] = id
        Grid_Type[pos] = CellType.FCell

        # Find the fragments for the FCell
        fcell_id = id
        fragments = cell.Fragments
        x, y, z = pos
        for i in range(1, DENDRITE_LENGTH + 1):
            for frag_pos in [(x + i, y, z), (x - i, y, z), (x, y + i, z), (x, y - i, z), (x, y, z - i)]:
                if (frag_pos[0] - (N / 2 + 0.5)) ** 2 + (frag_pos[1] - (N / 2 + 0.5)) ** 2 + (
                            frag_pos[2] - (N / 2 + 0.5)) ** 2 <= (N / 2) ** 2 and Grid_ID[frag_pos] is None:
                    id = available_cell_ids.pop()
                    fragments.append(id)
                    cell = SimpleNamespace(Type=CellType.Fragment, Position=frag_pos, antigenAmount=None, icAmount=0,
                                           Parent=fcell_id)
                    cell_properties[id] = cell
                    Grid_ID[frag_pos] = id
                    Grid_Type[frag_pos] = CellType.Fragment

            # When Z axis is increasing, we require an extra check to ensure that we're still in light zone.
            frag_pos = (x, y, z + i)
            if (frag_pos[0] - (N / 2 + 0.5)) ** 2 + (frag_pos[1] - (N / 2 + 0.5)) ** 2 + (
                        frag_pos[2] - (N / 2 + 0.5)) ** 2 <= (N / 2) ** 2 and frag_pos[2] <= N // 2 and Grid_ID[
                frag_pos] is None:
                id = available_cell_ids.pop()
                fragments.append(id)
                cell = SimpleNamespace(Type=CellType.Fragment, Position=frag_pos, antigenAmount=None, icAmount=0.0,
                                       Parent=fcell_id)
                cell_properties[id] = cell
                Grid_ID[frag_pos] = id
                Grid_Type[frag_pos] = CellType.Fragment

        # Assign each fragment an amount of antigen
        fcell_vol = len(fragments) + 1  # +1 accounts for centre
        ag_per_frag = INITIAL_ANTIGEN_AMOUNT_PER_FDC / fcell_vol
        for id in [fcell_id] + fragments:
            cell = cell_properties[id]
            cell.antigenAmount = ag_per_frag

    # Initialise Seeder Cells:
    for _ in range(INITIAL_NUM_SEEDER):
        pos = random.choice(LIGHT_ZONE)  # Find empty location in list zone
        while Grid_ID[pos] is not None:
            pos = random.choice(LIGHT_ZONE)

        # Obtain new ID for new cell
        id = available_cell_ids.pop()

        # Add cell to appropriate lists and dictionaries
        list_cb.append(id)
        polarity_vector = np.random.standard_normal(3)
        polarity_vector = polarity_vector / np.linalg.norm(polarity_vector)
        cell = SimpleNamespace(Type=CellType.Centroblast, Position=pos, State=None,
                               BCR=random.choice(tuple(BCR_values_all)), Polarity=polarity_vector,
                               responsiveToCXCL12=None,
                               responsiveToCXCL13=None, numDivisionsToDo=NUM_DIV_INITIAL_CELLS, pMutation=p_mut(0.0),
                               IAmHighAg=False, retainedAg=0.0,
                               cycleStartTime=None, endOfThisPhase=None)
        cell_properties[id] = cell

        cell_properties = initiate_cycle(id, cell_properties)
        cell_properties = initiate_chemokine_receptors(id, cell_properties)

    # Initialise T Cells:
    for i in range(INITIAL_NUM_TC):
        pos = random.choice(LIGHT_ZONE)  # Find empty location in light zone
        while Grid_ID[pos] is not None:
            pos = random.choice(LIGHT_ZONE)

        # Obtain new ID for new cell.
        id = available_cell_ids.pop()

        # Add cell to appropriate lists and dictionaries
        list_tc.append(id)
        polarity_vector = np.random.standard_normal(3)
        polarity_vector = polarity_vector / np.linalg.norm(polarity_vector)
        cell = SimpleNamespace(Type=CellType.TCell, Position=pos, State=CellState.TCnormal, Polarity=polarity_vector,
                               responsiveToCXCL12=False, responsiveToCXCL13=False, BCell_Contacts=[])
        cell_properties[id] = cell
        Grid_ID[pos] = id
        Grid_Type[pos] = CellType.TCell

    return available_cell_ids, cell_properties, list_stromal, list_fdc, list_cb, list_cc, list_tc, list_outcells


def hyphasma():
    """
    Algorithm 10, Simulation of Germinal Center.
    Main driver function for the simulation of a Germinal Center.
    :return:
    """
    # Dictionaries storing amounts of CXCL12 and CXCL13 at each point:

    grid_cxcl12 = {pos: random.uniform(80e-11, 80e-10) for pos in
                   [(x + N / 2 + 1, y + N / 2 + 1, z + N / 2 + 1) for x in range(-N // 2 - 1, N // 2 + 1) for y in
                    range(-N // 2 - 1, N // 2 + 1)
                    for z in range(-N // 2 - 1, N // 2 + 1)]}
    grid_cxcl13 = {pos: random.uniform(0.1e-10, 0.1e-9) for pos in
                   [(x + N / 2 + 1, y + N / 2 + 1, z + N / 2 + 1) for x in range(-N // 2 - 1, N // 2 + 1) for y in
                    range(-N // 2 - 1, N // 2 + 1)
                    for z in range(-N // 2 - 1, N // 2 + 1)]}

    available_cell_ids, cell_properties, list_stromal, list_fdc, list_cb, list_cc, list_tc, list_outcells = initialise_cells()

    # For plots/tests:
    num_bcells = []
    times = []

    t = 0.0
    while t <= TMAX:
        # Some of the following lines are purely for checking the simulation is running alright.
        t = round(t,3)
        print(t)
        # Track the number of B cellsat each time step.
        num_bcells.append(len(list_cc) + len(list_cb))
        if num_bcells[-1] > 3:
            print("Number B Cells: {}".format(num_bcells[-1]))
            print("Number Centroblasts: {}".format(len(list_cb)))
            print("Number Centrocytes: {}".format(len(list_cc)))
        times.append(t)

        # Secrete CXCL12 from Stromal Cells.
        for id in list_stromal:
            signal_secretion(id, 'CXCL12', p_mkCXCL12)

        # Randomly iterate over F cells / Fragments.
        random.shuffle(list_fdc)
        for id in list_fdc:
            cell = cell_properties[id]
            # Secrete CXCL13 from Fcells
            signal_secretion(id, 'CXCL13', p_mkCXCL13)
            fragments = cell.Fragments
            # Update antigen amounts for each fragment.
            for frag_id in fragments:
                frag = cell_properties[frag_id]
                for bcr_seq in BCR_values_all:
                    d_ic = DT * (
                        K_ON * frag.antigenAmount * AntibodyPerBCR[bcr_seq] - k_off(bcr_seq) * frag.icAmount)
                    frag.antigenAmount -= d_ic
                    frag.icAmount += d_ic

        # Update the number of outcells and amount of antibody for each CR value.
        for bcr_seq in BCR_values_all:
            transfert = math.floor(NumBCROutCells[bcr_seq] * PM_DIFFERENTIATION_RATE * DT)
            NumBCROutCells[bcr_seq] -= transfert
            NumBCROutCellsProduce[bcr_seq] += transfert
            AntibodyPerBCR[bcr_seq] = NumBCROutCellsProduce[bcr_seq] * AB_PROD_FACTOR - ANTIBODY_DEGRADATION * \
                                                                                        AntibodyPerBCR[bcr_seq]
        # Randomly iterate over Outcells
        random.shuffle(list_outcells)
        for id in list_outcells:
            cell = cell_properties[id]
            # Move cell and remove if on surface of sphere / Germinal Center.
            cell_properties = move(id, cell_properties, grid_cxcl12, grid_cxcl13)
            pos = cell.Position
            if is_surface_point(pos):
                list_outcells.remove(id)
                available_cell_ids.append(id)

        # Randomly iterate over Centroblast cells.
        random.shuffle(list_cb)
        for id in list_cb:
            cell = cell_properties[id]
            # Progress cells in their lifetime cycle.
            cell_properties = update_chemokines_receptors(id, cell_properties, grid_cxcl12, grid_cxcl13)
            cell_properties = progress_cycle(id, cell_properties)
            # Attempt to divide cell if ready.
            if cell.State == CellState.cb_divide:
                cell_properties, available_cell_ids, list_cb = divide_and_mutate(id, cell_properties,
                                                                                 available_cell_ids, t, list_cb)
            if cell.State == CellState.Stop_Dividing:
                if random.uniform(0, 1) < PROB_DIF:
                    if cell.IAmHighAg:
                        cell_properties, list_outcells, list_cb = differ_to_out(id, cell_properties, list_outcells,
                                                                                list_cb)
                    else:
                        cell_properties, list_cc, list_cb = differ_to_cc(id, cell_properties, list_cc, list_cb)

            # Move allowed cells.
            if cell.State != CellState.cb_M:
                cell_properties = move(id, cell_properties, grid_cxcl12, grid_cxcl13)

        # Randomly iterate over Centrocyte cells.
        random.shuffle(list_cc)
        for id in list_cc:
            cell = cell_properties[id]
            # Update cells progress
            cell_properties = update_chemokines_receptors(id, cell_properties, grid_cxcl12, grid_cxcl13)
            cell_properties = progress_fdc_selection(id, cell_properties)
            cell_properties, list_cc, list_outcells, list_cb = progress_tcell_selection(id, cell_properties, t, list_cc,
                                                                                        list_outcells, list_cb)
            # Remove cell from simulation if dead or move if not in contact with T or F cell.
            if cell.State == CellState.Apoptosis:
                list_cb.remove(id)
                available_cell_ids.append(id)
            elif cell.State not in [CellState.FDCcontact, CellState.TCcontact]:
                cell_properties = move(id, cell_properties, grid_cxcl12, grid_cxcl13)

        # Randomly iterate over T cells and move if not attached to another cell.
        random.shuffle(list_tc)
        for id in list_tc:
            cell = cell_properties[id]
            if cell.State == CellState.TCnormal:
                cell_properties = move(id, cell_properties, grid_cxcl12, grid_cxcl13)

        # At this point we can add in more B cells using a similar process to algorithm 9.

        # TODO collision resolution, lines 68-72
        t += DT
    return times, num_bcells


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
    hamming_dist = sum(el1 != el2 for el1, el2 in zip(str(bcr), str(ANTIGEN_VALUE)))
    return K_ON / (10 ** (EXP_MIN + math.exp(-(hamming_dist / 2.8) ** 2) * (EXP_MAX - EXP_MIN)))


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
    mu = mu/3
    result = random.gauss(mu, sigma)
    while result<0:
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
        if (neighbour_pos[0] - (N / 2 + 0.5)) ** 2 + (neighbour_pos[1] - (N / 2 + 0.5)) ** 2 + (
                    neighbour_pos[2] - (N / 2 + 0.5)) ** 2 > (N / 2) ** 2:
            surface = True
    return surface


# Set-up for simulation:
ANTIGEN_VALUE = 1234

# Distance Variables:
N = 16  # Diameter of sphere/GC
ALL_POINTS = [(x + N // 2 + 1, y + N // 2 + 1, z + N // 2 + 1) for x in range(-N // 2, N // 2) for y in
              range(-N // 2, N // 2)
              for z in range(-N // 2, N // 2) if ((x + 0.5) ** 2 + (y + 0.5) ** 2 + (z + 0.5) ** 2) <= (N // 2) ** 2]
DARK_ZONE = [point for point in ALL_POINTS if point[2] > N // 2]
LIGHT_ZONE = [point for point in ALL_POINTS if point[2] <= N // 2]
OFFSET = (N / 2 + 0.5, N / 2 + 0.5, N / 2 + 0.5)

# Spatial step size (micrometers)
DX = 5

# Time Variables:
DT = 0.002
TMIN = 0.0
TMAX = 30.0

# Initialisation
INITIAL_NUM_STROMAL_CELLS = 30
INITIAL_NUM_FDC = 20
INITIAL_NUM_SEEDER = 3
INITIAL_NUM_TC = 25

DENDRITE_LENGTH = 8
INITIAL_ANTIGEN_AMOUNT_PER_FDC = 3000

# Possible initial BCR values
BCR_values_initial = random.sample(range(1000, 10000), 1000)
BCR_values_all = set(BCR_values_initial)
NumBCROutCells = {bcr: 0 for bcr in BCR_values_initial}
NumBCROutCellsProduce = {bcr: 0 for bcr in BCR_values_initial}
AntibodyPerBCR = {bcr: 0 for bcr in BCR_values_initial}

# Dictionaries storing what is at each location. Initially empty, so 'None'.
Grid_ID = {pos: None for pos in ALL_POINTS}
Grid_Type = {pos: None for pos in ALL_POINTS}

# Dynamic number of divisions:
NUM_DIV_INITIAL_CELLS = 3
pMHCdepHill = 1.0
pMHCdepMin = 1.0
pMHCdepMax = 6.0
pMHCdepK = 6.0

# Production/ Diffusion Rates:
p_mkCXCL12 = 4e-7
p_mkCXCL13 = 1e-8
CXCL13_DiffRate = 1000 * 25 * 0.002

# Persistent Length time (PLT)
PLT_CENTROCYTE = 0.025
PLT_CENTROBLAST = 0.025
PLT_TCELL = 0.0283
PLT_OUTCELL = 0.0125

# Dynamic update of chemokine receptors
CXCL12_CRIT = 60.0e-10
CXCL12_RECRIT = 40.0e-10
CXCL13_CRIT = 0.8e-10
CXCL13_RECRIT = 0.6e-10

# Chemotaxis
CHEMO_MAX = 10
CHEMO_STEEP = 1e+10
CHEMO_HALF = 2e-11
NORTH_WEIGHT = 0.1
NORTH = np.array([0, 0, -1])

# Speed
SPEED_CENTROCYTE = 7.5
SPEED_CENTROBLAST = 7.5
SPEED_TCELL = 10.0
SPEED_OUTCELL = 3.0

# Divide and Mutate
PROB_NOW = DT * 9.0 * 10
MUTATION_START_TIME = 2.0
POLARITY_INDEX = 0.88
PROB_DIVIDE_AG_ASYMMETRIC = 0.72
PROB_MUT_AFTER_SELECTION = 0.0
PROB_MUT_AFFINITY_EXPONENT = 1.0

# Differentiation Rates
START_DIFFERENTIATION = 72.0
PROB_DIF = DT * 0.1
DELETE_AG_IN_FRESH_CC = True
DIF_DELAY = 6.0

PROB_DIF_TO_OUT = 0.0

# Selection Steps
TEST_DELAY = 0.02
COLLECT_FDC_PERIOD = 0.7
ANTIGEN_SATURATION = 20
P_SEL = DT * 0.05
TC_TIME = 0.6
TC_RESCUE_TIME = 0.5

# Antibody
PM_DIFFERENTIATION_RATE = 24.0
ANTIBODIES_PRODUCTION = 1e-17
V_BLOOD = 1e-2
N_GC = 1000
AB_PROD_FACTOR = ANTIBODIES_PRODUCTION * DT * N_GC * V_BLOOD * 1e15
ANTIBODY_DEGRADATION = 30
K_ON = 3.6e9

EXP_MIN = 5.5
EXP_MAX = 9.5

# Movements:
POSSIBLE_NEIGHBOURS = list(itertools.product([-1, 0, 1], repeat=3))
POSSIBLE_NEIGHBOURS.remove((0, 0, 0))


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
    times, num_bcells = hyphasma()
    plt.plot(times, num_bcells)
    plt.show()
