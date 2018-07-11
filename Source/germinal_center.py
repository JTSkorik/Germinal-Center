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

# TODO signal_secretion, diffusion, and turn_angle functions.

# noinspection PyUnresolvedReferences
def mutate(ID):
    """
    Algorithm 1, Mutation.
    Mutates the BCR value for a cell with a certain probability.
    :param ID: integer, determines which cell in the pop we are talking about.
    :return:
    """
    cell = Cells[ID]
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
    if cell.BCR not in BCR_values_all:
        BCR_values_all.add(cell.BCR)
        NumBCROutCells[cell.BCR] = 0
        NumBCROutCellsProduce[cell.BCR] = 0
        AntibodyPerBCR[cell.BCR] = 0


# noinspection PyUnresolvedReferences
def initiate_chemokine_receptors(ID):
    """
    Algorithm 2, Dynamic Updating of Chemotaxis.
    Used to determine the attributes responsive_to_CXCL12 and responsive_to_CXCL13 of a cell given its type.
    :param ID: integer, determines which cell in the pop we are talking about.
    :return:
    """
    cell = Cells[ID]
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


# noinspection PyUnresolvedReferences
def update_chemokines_receptors(ID):
    """
    Algorithm 2, Dynamic Updating of Chemotaxis.
    Used to update the attributes responsive_to_CXCL12 and responsive_to_CXCL13 of a cell given its type and state.
    :param ID: integer, determines which cell in the pop we are talking about.
    :return:
    """
    cell = Cells[ID]
    pos = cell.Position
    if cell.Type == CellType.Centrocyte:
        if cell.State == CellState.unselected:
            if Grid_CXCL13[pos] > CXCL13crit:
                cell.responsiveToCXCL13 = False
            elif Grid_CXCL13[pos] < CXCL13recrit:
                cell.responsiveToCXCL13 = True
    elif cell.Type == CellType.Centroblast:
        if Grid_CXCL12[pos] > CXCL12crit:
            cell.responsiveToCXCL12 = False
        elif Grid_CXCL12[pos] < CXCL12recrit:
            cell.responsiveToCXCL12 = True


# noinspection PyUnresolvedReferences,PyUnboundLocalVariable
def move(ID):
    # TODO add check that the cells are staying inside their respective zones.
    """
    Algorithm 3, Updating Position and Polarity of cells at each time-point.
    Updates the polarity of a cell and then will move the cell within the Germinal center. Both events occur stochastically.
    :param ID: integer, determines which cell in the pop we are talking about.
    :return:
    """
    cell = Cells[ID]
    # Obtain current position of cell.
    pos = cell.Position
    x = pos[0]
    y = pos[1]
    z = pos[2]

    # Obtain required parameters
    cell_type = cell.Type
    if cell_type == CellType.Centrocyte:
        prob = pLTCentrocyte
        speed = speedCentrocyte
    elif cell_type == CellType.Centroblast:
        prob = pLTCentroblast
        speed = speedCentroblast
    elif cell_type == CellType.TCell:
        prob = pLTTCell
        speed = speedTCell
    elif cell_type == CellType.OutCell:
        prob = pLTOutCell
        speed = speedOutCell
    else:
        prob = None
        speed = None
        print("move: Invalid cell_type, {}".format(cell_type))

    # Calculate new polarity
    if random.uniform(0, 1) < prob:
        # Turning angles influence
        theta = random.gauss(0, 1)
        phi = random.uniform(0, 2 * math.pi)
        turn_angle(ID, theta, phi)

        # Influence due to CXCL12
        if cell.responsiveToCXCL12:
            x_diff = Grid_CXCL12[(x + 1, y, z)] - Grid_CXCL12[(x - 1, y, z)]
            y_diff = Grid_CXCL12[(x, y + 1, z)] - Grid_CXCL12[(x, y - 1, z)]
            z_diff = Grid_CXCL12[(x, y, z + 1)] - Grid_CXCL12[(x, y, z - 1)]

            Gradient_CXCL12 = np.array([x_diff / (2 * dx), y_diff / (2 * dx), z_diff / (2 * dx)])

            mag_Gradient_CXCL12 = np.linalg.norm(Gradient_CXCL12)
            chemoFactor = (chemoMax / (
                1 + math.exp(chemoSteep * (chemoHalf - 2 * dx * mag_Gradient_CXCL12)))) * Gradient_CXCL12

            cell.Polarity += chemoFactor

        # Influence due to CXCL13
        if cell.responsiveToCXCL13:
            x_diff = Grid_CXCL13[(x + 1, y, z)] - Grid_CXCL13[(x - 1, y, z)]
            y_diff = Grid_CXCL13[(x, y + 1, z)] - Grid_CXCL13[(x, y - 1, z)]
            z_diff = Grid_CXCL13[(x, y, z + 1)] - Grid_CXCL13[(x, y, z - 1)]

            Gradient_CXCL13 = np.array([x_diff / (2 * dx), y_diff / (2 * dx), z_diff / (2 * dx)])
            mag_Gradient_CXCL13 = np.linalg.norm(Gradient_CXCL13)
            chemoFactor = (chemoMax / (
                1 + math.exp(chemoSteep * (chemoHalf - 2 * dx * mag_Gradient_CXCL13)))) * Gradient_CXCL13

            cell.Polarity += chemoFactor

        # Influence specific to T Cells
        if cell_type == CellType.TCell:
            cell.Polarity = (1.0 - northweight) * cell.Polarity + northweight * north

        cell.Polarity = cell.Polarity / np.linalg.norm(cell.Polarity)

    # Probability of movement
    pDifu = speed * dt / dx

    if random.uniform(0, 1) < pDifu:
        # Find possible new positions based in order of best preference
        WantedPosition = np.asarray(pos) + cell.Polarity
        Neighbours = [np.asarray(Movement) + np.asarray(pos) for Movement in
                      Possible_Movements if np.linalg.norm(
                np.asarray(Movement) + np.asarray(pos) - np.array(OffSet)) <= (N / 2)]
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


def turn_angle(ID, theta, phi):
    """
    Algorithm 3 (Updating Position and Polarity of cells at each time-point)
    Rotates the polarity of a cell by the given angles.
    Yet to be finished.
    :param ID: integer, determines which cell in the pop we are talking about.
    :param theta: float, randomly generated turning angle.
    :param phi: float, another randomly generated turning angle between 0 and 2pi.
    :return:
    """
    cell = Cells[ID]

    v = np.random.standard_normal(3)
    cell.Polarity = v / np.linalg.norm(v)


def initiate_cycle(ID):
    """
    Algorithm 3, Updating events at the Centroblast Stage.
    Sets the state for a cell depending on how many divisions left to complete.
    :param ID: integer, determines which cell in the pop we are talking about.
    :return:
    """

    cell = Cells[ID]
    # noinspection PyUnresolvedReferences
    if cell.numDivisionsToDo == 0:
        cell.State = CellState.Stop_Dividing
    else:
        cell.State = CellState.cb_G1
        cell.cycleStartTime = 0
        cell.endOfThisPhase = get_duration(ID)


def progress_cycle(ID):
    """
    Algorithm 4, Updating events at the Centroblast Stage.
    Progresses the cell to its next state and calculates how long it till next change of state.
    :param ID: integer, determines which cell in the pop we are talking about.
    :return:
    """
    cell = Cells[ID]
    # Progress cell into its next state
    cell.cycleStartTime += dt
    if cell.cycleStartTime > cell.endOfThisPhase:
        if cell.State == CellState.cb_G1:
            cell.State = CellState.cb_S
        elif cell.State == CellState.cb_S:
            cell.State = CellState.cb_G2
        elif cell.State == CellState.cb_G2:
            cell.State = CellState.cb_divide

        # Finds time till end of new state.
        if cell.State != CellState.cb_divide:
            cell.endOfThisPhase = get_duration(ID)
            cell.cycleStartTime = 0


# noinspection PyUnresolvedReferences
def divide_and_mutate(ID, t):
    """
    Algorithm 4, Updating events at the Centroblast Stage.
    Takes a cell and divides it with probability pNow. Will also attempt to mutate the cells.
    :param ID: integer, determines which cell in the pop we are talking about.
    :param t: float, current time of the simulation.
    :return:
    """
    old_cell = Cells[ID]
    if random.uniform(0, 1) < pNow:
        # Find all empty positions neighbouring cell.
        pos = old_cell.Position
        empty_neighbours = [tuple(np.array(pos) + np.array(possible_neighbour)) for possible_neighbour in
                            Possible_Movements if np.linalg.norm(
                np.asarray(possible_neighbour) + np.asarray(pos) - np.array(OffSet)) <= (N / 2)]

        # Randomly choose one position for new cell
        if empty_neighbours:
            divide_pos = random.choice(empty_neighbours)

            # Generate a new ID for the cell and copy over the properties from the old cell.
            newID = AvailableCellIDs.pop()
            new_cell = copy.copy(old_cell)
            new_cell.Polarity = np.array(old_cell.Polarity)
            new_cell.Position = divide_pos
            new_cell.retainedAg = None
            Cells[newID] = new_cell

            Grid_ID[divide_pos] = newID
            Grid_ID[divide_pos] = Type[newID]

            old_cell.numDivisionsToDo -= 1
            new_cell.numDivisionsToDo -= 1

            old_cell.IAmHighAg = False
            new_cell.IAmHighAg = False

            # Initiate the cycle for each cell.
            initiate_cycle(ID)
            initiate_cycle(newID)

            # Mutate the cells
            if t > mutationStartTime:
                mutate(ID)
                mutate(newID)

            # Assign amount of retained antigen to each cell
            if random.uniform(0, 1) < pDivideAgAsymmetric:
                if old_cell.retainedAg == 0:
                    new_cell.retainedAg = 0
                else:
                    sep = random.gauss(polarityIndex, 1)
                    while sep < 0 or sep > 1:
                        sep = random.gauss(polarityIndex, 1)

                    new_cell.retainedAg = sep * old_cell.retainedAg
                    old_cell.retainedAg = (1 - sep) * old_cell.retainedAg

                    if sep > 0.5:
                        new_cell.IAmHighAg = True
                    else:
                        old_cell.IAmHighAg = False
            else:
                new_cell.retainedAg = old_cell.retainedAg / 2
                old_cell.retainedAg = old_cell.retainedAg / 2


# noinspection PyUnresolvedReferences
def progress_fdc_selection(ID):
    """
    Algorithm 5, Antigen Collection from FDCs.
    Allows for centrocytes to collect antigen from neighbouring F cells / fragments.
    :param ID: integer, determines which cell in the pop we are talking about.
    :return:
    """
    cell = Cells[ID]
    if cell.State == CellState.Unselected:
        # Progress selected clock and check if able to collect antigen.
        cell.selectedClock += dt
        if cell.selectedClock <= collectFDCperiod:
            cell.Clock += dt
            # noinspection PyTypeChecker
            if cell.Clock > testDelay:
                cell.selectable = True
                # Find frag component with largest amount of antigen.
                Frag_max = None
                Frag_max_ID = None
                pos = cell.Position
                for neighbour in Possible_Movements:
                    neighbour_pos = tuple(np.array(pos) + np.array(neighbour))
                    if Grid_Type[neighbour_pos] in [CellType.Fragment, CellType.FCell]:
                        Frag_ID = Grid_ID[neighbour_pos]
                        Frag_cell = Cells[Frag_ID]
                        if Frag_cell.antigenAmount > Frag_max:
                            Frag_max = Frag_cell.antigenAmount
                            Frag_max_ID = Frag_ID

                pBind = affinity(ID) * Frag_max / antigenSaturation
                # Bind cell and fragment with probability pBind or reset clock.
                if random.choice(0, 1) < pBind:
                    cell.State = CellState.FDCcontact
                    cell.FragContact = Frag_max_ID
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
        cell.selectedClock += dt
        if random.uniform(0, 1) < pSel:
            cell.numFDCContacts += 1
            Frag_ID = cell.FragContact
            Frag_cell = Cells[Frag_ID]
            Frag_cell.antigenAmount -= 1
            cell.State = CellState.Unselected
            cell.Clock = 0
            cell.selectable = False


# noinspection PyUnresolvedReferences,PyTypeChecker,PyTypeChecker
def progress_tcell_selection(ID, t):
    """
    Algorithm 6, Screening for T cell help at Centrocyte Stage.

    :param ID: integer, determines which cell in the pop we are talking about.
    :param t: t: float, current time of the simulation.
    :return:
    """
    cell = Cells[ID]
    if cell.State == CellState.FDCselected:
        # Find if there is a neighbouring T cell.
        pos = cell.Position
        for neighbour in Possible_Movements:
            neighbour_pos = tuple(np.array(pos) + np.array(neighbour))
            # If there is a neighbouring T cell, we record the contact.
            if Grid_Type[neighbour_pos] == CellType.TCell and cell.State != CellState.TCcontact:
                update_tcell(ID, Grid_ID[neighbour_pos])
                cell.State = CellState.TCcontact
                cell.tcClock = 0
                cell.tcSignalDuration = 0

    elif cell.State == CellState.TCcontact:
        cell.tcClock += dt
        # Check is current cell has least amount of antigens compared to T cells neighbouring cells.
        lowest_antigen = True
        for ID_BC in TCellInteractions[BCellInteractions[ID]]:
            other_cell = Cells[ID_BC]
            if ID != ID_BC and cell.retainedAg <= other_cell.retainedAg:
                lowest_antigen = False
        if lowest_antigen:
            cell.tcSignalDuration += dt
        if cell.tcSignalDuration > tcRescueTime:
            cell.State = CellState.Selected
            cell.selectedClock = 0
            rand = random.uniform(0, 1)
            cell.IndividualDifDelay = dif_delay * (1 + 0.1 * math.log(1 - rand) / rand)
            liberate_tcell(ID, BCellInteractions[ID])
        elif cell.tcClock >= tcTime:
            cell.State = CellState.Apoptosis
            liberate_tcell(ID, BCellInteractions[ID])

    elif cell.State == CellState.Selected:
        cell.selectedClock += dt
        if cell.selectedClock > cell.IndividualDifDelay:
            if random.uniform(0, 1) < pDif:
                if random.uniform(0, 1) < pDifToOut:
                    differ_to_out(ID)
                else:
                    differ_to_cb(ID, t)


def update_tcell(ID_B, ID_T):
    """
    Algorithm 7, Updating the T cells according to B cells interactions.
    Updates records of whether cells are interacting, specifically marking two cells as interacting.
    :param ID_B: integer, determines which B cell in the pop we are talking about.
    :param ID_T: integer, determines which T cell in the pop we are talking about.
    :return:
    """
    T_Cell = Cells[ID_T]
    TCellInteractions[ID_T].append(ID_B)
    BCellInteractions[ID_B] = ID_T
    T_Cell.State = CellState.TC_CC_Contact


def liberate_tcell(ID_B, ID_T):
    """
    Algorithm 7, Updating the T cells according to B cells interactions.
    Updates records of whether cells are interacting, specifically removing cells from being interactive.
    :param ID_B: integer, determines which B cell in the pop we are talking about.
    :param ID_T: integer, determines which T cell in the pop we are talking about.
    :return:
    """
    T_Cell = Cells[ID_T]
    TCellInteractions[ID_T].remove(ID_B)
    BCellInteractions[ID_B] = None
    if not TCellInteractions[ID_T]:
        T_Cell.State = CellState.TCnormal


# noinspection PyUnresolvedReferences
def differ_to_out(ID):
    """
    Algorithm 8, Transition between Centroblasts, Centrocyctes, and Output Cells.
    Transition a cell from Centroblast or Centrocycte to an Output Cell.
    :param ID: integer, determines which cell in the pop we are talking about.
    :return:
    """
    OutList.append(ID)
    old_cell = Cells[ID]
    NumBCROutCells[old_cell.BCR] += 1
    # Update cell and Grid position properties.
    new_cell = SimpleNamespace(Type=CellType.OutCell, Position=old_cell.Position, Polarity=old_cell.Polarity,
                               responsiveToCXCL12=None, responsiveToSignalCXCL13=None)
    Cells[ID] = new_cell
    Grid_Type[cell.Position] = CellType.OutCell
    initiate_chemokine_receptors(ID)


# noinspection PyUnresolvedReferences
def differ_to_cb(ID, t):
    """
    Algorithm 8, Transition between Centroblasts, Centrocyctes, and Output Cells.
    Transitions a cell from Centrocyte to Centroblast.
    :param ID: integer, determines which cell in the pop we are talking about.
    :param t: float,  current time of the simulation.
    :return:
    """
    CBList.append(ID)
    old_cell = Cells[ID]

    # Update grid and cell details.
    Grid_Type[old_cell.Position] = CellType.Centroblast
    new_cell = SimpleNamespace(Type=CellType.Centroblast, Position=old_cell.Position, State=None, BCR=old_cell.BCR,
                               Polarity=old_cell.Polarity, responsiveToCXCL12=None,
                               responsiveToCXCL13=None, numDivisionsToDo=None, pMutation=None, IAmHighAg=True,
                               retainedAg=old_cell.NumFDCContacts, cycleStartTime=None, endOfThisPhase=None)

    Cells[ID] = new_cell
    # Find number of divisions to do.
    agFactor = old_cell.numFDCContacts ** pMHCdepHill
    new_cell.numDivisionsToDo = pMHCdepMin + (pMHCdepMax - pMHCdepMin) * agFactor / (agFactor + pMHCdepK ** pMHCdepHill)

    # Find new probability of mutation.
    new_cell.pMutation = p_mut(t) + (pMutAfterSelection - p_mut(t)) * affinity(ID) ** pMutAffinityExponent

    # Initiate cell.
    initiate_chemokine_receptors(ID)
    initiate_cycle(ID)


# noinspection PyUnresolvedReferences
def differ_to_cc(ID):
    """
    Algorithm 8, Transition between Centroblasts, Centrocyctes, and Output Cells.
    Transitions a cell from Centroblast to Centrocyte.
    :param ID: integer, determines which cell in the pop we are talking about.
    :return:
    """
    CCList.append(ID)
    old_cell = Cells[ID]

    new_cell = SimpleNamespace(Type=CellType.Centrocyte, Position=old_cell.Position, State=CellState.Unselected,
                               BCR=old_cell.BCR, Polarity=old_cell.Polarity, responsiveToCXCL12=None,
                               responsiveToCXCL13=None, selectedClock=0, Clock=0, selectable=False, FragContact=None,
                               numFDCContacts=None, tcClock=None, tcSignalDuration=None, IndividualDifDelay=None)
    Cells[ID] = new_cell
    initiate_chemokine_receptors(ID)
    Grid_Type[new_cell.Position] = CellType.Centrocyte
    if DeleteAgInFreshCC:
        new_cell.numFDCContacts = 0
    else:
        new_cell.numFDCContacts = math.floor(old_cell.retainedAg + 0.5)


def initialise_cells():
    """
    Algorithm 9, Initialisation.
    Starts the simulation by introducing various amounts of the possible cells into the simulation.
    :return:
    """
    # Initialise Stromal Cells:
    for _ in range(NumStromalCells):
        # Find empty location in dark zone
        pos = random.choice(DarkZone)
        while Grid_ID[pos] is not None:
            pos = random.choice(DarkZone)

        # Obtain new ID for new cell
        ID = AvailableCellIDs.pop()

        # Add to appropriate lists and dictionaries
        StormaList.append(ID)
        cell = SimpleNamespace(Type=CellType.Stromal, Position=pos)
        Cells[ID] = cell
        Grid_ID[pos] = ID
        Grid_Type[pos] = CellType.Stromal

    # Initialise Fragments:
    for _ in range(NumFDC):
        # Find empty location in light zone
        pos = random.choice(LightZone)
        while Grid_ID[pos] is not None:
            pos = random.choice(LightZone)

        # Obtain new ID for new cell
        ID = AvailableCellIDs.pop()

        # Add to appropriate lists and dictionaries
        FDCList.append(ID)
        cell = SimpleNamespace(Type=CellType.FCell, Position=pos, antigenAmount=None, icAmount=0, Fragments=[])
        Cells[ID] = cell
        Grid_ID[pos] = ID
        Grid_Type[pos] = CellType.FCell

        # Find the fragments for the FCell
        FCell_ID = ID
        fragments = cell.Fragments
        x = pos[0]
        y = pos[1]
        z = pos[2]
        for i in range(1, DendriteLength + 1):
            for frag_pos in [(x + i, y, z), (x - i, y, z), (x, y + i, z), (x, y - i, z), (x, y, z - i)]:
                if (frag_pos[0] - (N / 2 + 0.5)) ** 2 + (frag_pos[1] - (N / 2 + 0.5)) ** 2 + (
                            frag_pos[2] - (N / 2 + 0.5)) ** 2 <= (N / 2) ** 2 and Grid_ID[frag_pos] is None:
                    ID = AvailableCellIDs.pop()
                    fragments.append(ID)
                    cell = SimpleNamespace(Type=CellType.Fragment, Position=frag_pos, antigenAmount=None, icAmount=0,
                                           Parent=FCell_ID)
                    Cells[ID] = cell
                    Grid_ID[frag_pos] = ID
                    Grid_Type[frag_pos] = CellType.Fragment

            # When Z axis is increasing, we require an extra check to ensure that we're still in light zone.
            frag_pos = (x, y, z + i)
            if (frag_pos[0] - (N / 2 + 0.5)) ** 2 + (frag_pos[1] - (N / 2 + 0.5)) ** 2 + (
                        frag_pos[2] - (N / 2 + 0.5)) ** 2 <= (N / 2) ** 2 and frag_pos[2] <= N // 2 and Grid_ID[
                frag_pos] is None:
                ID = AvailableCellIDs.pop()
                fragments.append(ID)
                cell = SimpleNamespace(Type=CellType.Fragment, Position=frag_pos, antigenAmount=None, icAmount=0,
                                       Parent=FCell_ID)
                Cells[ID] = cell
                Grid_ID[frag_pos] = ID
                Grid_Type[frag_pos] = CellType.Fragment

        # Assign each fragment an amount of antigen
        FCellVol = len(fragments) + 1  # +1 accounts for centre
        agPerFrag = AntigenAmountPerFDC / FCellVol
        for ID in [FCell_ID] + fragments:
            cell = Cells[ID]
            cell.antigenAmount = agPerFrag

    # Initialise Seeder Cells:
    for _ in range(NumSeeder):
        pos = random.choice(LightZone)  # Find empty location in list zone
        while Grid_ID[pos] is not None:
            pos = random.choice(LightZone)

        # Obtain new ID for new cell
        ID = AvailableCellIDs.pop()

        # Add cell to appropriate lists and dictionaries
        CBList.append(ID)
        v = np.random.standard_normal(3)
        v = v / np.linalg.norm(v)
        cell = SimpleNamespace(Type=CellType.Centroblast, Position=pos, State=None,
                               BCR=random.choice(tuple(BCR_values_all)), Polarity=v, responsiveToCXCL12=None,
                               responsiveToCXCL13=None, numDivisionsToDo=numDivFounderCells, pMutation=p_mut(0.0),
                               IAmHighAg=False, retainedAg=0,
                               cycleStartTime=None, endOfThisPhase=None)
        Cells[ID] = cell

        initiate_cycle(ID)
        initiate_chemokine_receptors(ID)

    # Initialise T Cells:
    for i in range(NumTC):
        pos = random.choice(LightZone)  # Find empty location in light zone
        while Grid_ID[pos] is not None:
            pos = random.choice(LightZone)

        # Obtain new ID for new cell.
        ID = AvailableCellIDs.pop()

        # Add cell to appropriate lists and dictionaries
        TCList.append(ID)
        v = np.random.standard_normal(3)
        v = v / np.linalg.norm(v)
        cell = SimpleNamespace(Type=CellType.TCell, Position=pos, State=CellState.TCnormal, Polarity=v,
                               responsiveToCXCL12=False, responsiveToCXCL13=False)
        Cells[ID] = cell
        Grid_ID[pos] = ID
        Grid_Type[pos] = CellType.TCell
        TCellInteractions[ID] = []


# noinspection PyUnresolvedReferences
def hyphasma():
    """
    Algorithm 10, Simulation of Germinal Center.
    Main driver function for the simulation of a Germinal Center.
    :return:
    """
    t = 0.0
    while t <= tmax:
        print(t)
        # Track the number of B cellsat each time step.
        NumBCells.append(len(CCList) + len(CBList))
        times.append(t)

        # Secrete CXCL12 from Stromal Cells.
        for ID in StormaList:
            signal_secretion(ID, 'CXCL12', p_mkCXCL12)

        # Randomly iterate over F cells / Fragments.
        random.shuffle(FDCList)
        for ID in FDCList:
            cell = Cells[ID]
            # Secrete CXCL13 from Fcells
            signal_secretion(ID, 'CXCL13', p_mkCXCL13)
            fragments = cell.Fragments
            # Update antigen amounts for each fragment.
            for frag_ID in fragments:
                frag = Cells[frag_ID]
                for bcr_seq in BCR_values_all:
                    d_ic = dt * (
                        k_on * frag.antigenAmount * AntibodyPerBCR[bcr_seq] - k_off(bcr_seq) * frag.icAmount)
                    frag.antigenAmount -= d_ic
                    frag.icAmount += d_ic

        # Update the number of outcells and amount of antibody for each CR value.
        for bcr_seq in BCR_values_all:
            transfert = math.floor(NumBCROutCells[bcr_seq] * pmDifferentiationRate * dt)
            NumBCROutCells[bcr_seq] -= transfert
            NumBCROutCellsProduce[bcr_seq] += transfert
            AntibodyPerBCR[bcr_seq] = NumBCROutCellsProduce[bcr_seq] * abProdFactor - antibodyDegradation * \
                                                                                      AntibodyPerBCR[bcr_seq]
        # Randomly iterate over Outcells
        random.shuffle(OutList)
        for ID in OutList:
            cell = Cells[ID]
            # Move cell and remove if on surface of sphere / Germinal Center.
            move(ID)
            pos = cell.Position
            if is_surface_point(pos):
                OutList.remove(ID)
                AvailableCellIDs.append(ID)

        # Randomly iterate over Centroblast cells.
        random.shuffle(CBList)
        for ID in CBList:
            cell = Cells[ID]
            # Progress cells in their lifetime cycle.
            update_chemokines_receptors(ID)
            progress_cycle(ID)
            # Attempt to divide cell if ready.
            if cell.State == CellState.cb_divide:
                divide_and_mutate(ID, t)
            if cell.State == CellState.Stop_Dividing:
                if random.uniform(0, 1) < pDif:
                    if cell.IAmHighAg:
                        differ_to_out(ID)
                        CBList.remove(ID)
                    else:
                        differ_to_cc(ID)
                        CBList.remove(ID)
            # Move allowed cells.
            if cell.State != CellState.cb_M:
                move(ID)

        # Randomly iterate over Centrocyte cells.
        random.shuffle(CCList)
        for ID in CCList:
            cell = Cells[ID]
            # Update cells progress
            update_chemokines_receptors(ID)
            progress_fdc_selection(ID)
            progress_tcell_selection(ID, t)
            # Remove cell from simulation if dead or move if not in contact with T or F cell.
            if cell.State == CellState.Apoptosis:
                CBList.remove(ID)
                AvailableCellIDs.append(ID)
            elif cell.State not in [CellState.FDCcontact, CellState.TCcontact]:
                move(ID)

        # Randomly iterate over T cells and move if not attached to another cell.
        random.shuffle(TCList)
        for ID in TCList:
            cell = Cells[ID]
            if cell.State == CellState.TCnormal:
                move(ID)

        # At this point we can add in more B cells using a similar process to algorithm 9.

        # TODO collision resolution, lines 68-72
        t += dt


# Extra Algorithms/Functions
def generate_spatial_points(n):
    """
    Calculates and finds the allowed positions (x,y,z) of the Germinal Center. Places them such that x, y, z in [0,n].
    :param n: interger, the number of discrete points across the diameter of the Germinal Center.
    :return: list of tuples, positions of all allowed points in the Germinal Center.
    """
    return [(x + n // 2 + 1, y + n // 2 + 1, z + n // 2 + 1) for x in range(-n // 2, n // 2) for y in
            range(-n // 2, n // 2)
            for z in range(-n // 2, n // 2) if ((x + 0.5) ** 2 + (y + 0.5) ** 2 + (z + 0.5) ** 2) <= (n // 2) ** 2]


def affinity(ID):
    """
    Calculates the affinity between the Antigen and a given cells BCR.
    :param ID: integer, determines which cell in the pop we are talking about.
    :return: float, affinity value.
    """
    cell = Cells[ID]
    # noinspection PyUnresolvedReferences
    hamming_dist = sum(el1 != el2 for el1, el2 in zip(str(cell.BCR), str(Antigen_Value)))
    return math.exp(-(hamming_dist / 2.8) ** 2)


def k_off(bcr):
    """
    Calulates k_off
    :param bcr: BCR value that has been exhibited throughout simulation.
    :return: float, value of k_off
    """
    hamming_dist = sum(el1 != el2 for el1, el2 in zip(str(bcr), str(Antigen_Value)))
    return k_on / (10 ** (expMin + math.exp(-(hamming_dist / 2.8) ** 2) * (expMax - expMin)))


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
def get_duration(ID):
    """
    Find duration of time before a cell divides. Amount of time determined using Guassian Random Variable.
    :param ID: integer, determines which cell in the pop we are talking about.
    :return: float, sample from a Guassian random variable representing time till a cell divides.
    """
    cell = Cells[ID]
    sigma = 1
    if cell.State == CellState.cb_G1:
        mu = 2.0
    elif cell.State == CellState.cb_S:
        mu = 1.0
    elif cell.State == CellState.cb_G2:
        mu = 2.5
    elif cell.State == CellState.cb_M:
        mu = 0.5
    else:
        mu = None
        print("getDuration: Invalid cell state, {}".format(cell.State))

    return random.gauss(mu, sigma)


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
Antigen_Value = 1234

# Distance Variables:
N = 16  # Diameter of sphere/GC
AllPoints = generate_spatial_points(N)
DarkZone = [point for point in AllPoints if point[2] > N // 2]
LightZone = [point for point in AllPoints if point[2] <= N // 2]
OffSet = (N / 2 + 0.5, N / 2 + 0.5, N / 2 + 0.5)

# Available Cell IDs:
AvailableCellIDs = list(range(len(AllPoints)))

# Numpy Array to store each cell's attributes
Cells = [None] * len(AllPoints)

# Spatial step size (micrometers)
dx = 5

# Time Variables:
dt = 0.002
tmin = 0.0
tmax = 20.0

# Initialisation
NumStromalCells = 30
NumFDC = 20
NumSeeder = 3
NumTC = 25

DendriteLength = 8
AntigenAmountPerFDC = 3000

# Lists to store ID of each cell in each state (and fragments)
StormaList = []
FDCList = []
CBList = []
CCList = []
TCList = []
OutList = []

# Possible initial BCR values
BCR_values_initial = random.sample(range(1000, 10000), 1000)
BCR_values_all = set(BCR_values_initial)
NumBCROutCells = {bcr: 0 for bcr in BCR_values_initial}
NumBCROutCellsProduce = {bcr: 0 for bcr in BCR_values_initial}
AntibodyPerBCR = {bcr: 0 for bcr in BCR_values_initial}

# Dictionaries storing what is at each location. Initially empty, so 'None'.
Grid_ID = {pos: None for pos in AllPoints}
Grid_Type = {pos: None for pos in AllPoints}

# Dictionaries storing amounts of CXCL12 and CXCL13 at each point:
# TODO needs refining.
Grid_CXCL12 = {pos: random.uniform(80e-11, 80e-10) for pos in
               [(x + N / 2 + 1, y + N / 2 + 1, z + N / 2 + 1) for x in range(-N // 2 - 1, N // 2 + 1) for y in
                range(-N // 2 - 1, N // 2 + 1)
                for z in range(-N // 2 - 1, N // 2 + 1)]}
Grid_CXCL13 = {pos: random.uniform(0.1e-10, 0.1e-9) for pos in
               [(x + N / 2 + 1, y + N / 2 + 1, z + N / 2 + 1) for x in range(-N // 2 - 1, N // 2 + 1) for y in
                range(-N // 2 - 1, N // 2 + 1)
                for z in range(-N // 2 - 1, N // 2 + 1)]}

# B cells interacting with T cells:
TCellInteractions = {}
BCellInteractions = {}

# Dynamic number of divisions:
numDivFounderCells = 12
pMHCdepHill = 1.0
pMHCdepMin = 1.0
pMHCdepMax = 6.0
pMHCdepK = 6.0

# Production/ Diffusion Rates:
p_mkCXCL12 = 4e-7
p_mkCXCL13 = 1e-8
CXCL13_DiffRate = 1000 * 25 * 0.002

# Persistent Length time
pLTCentrocyte = 0.025
pLTCentroblast = 0.025
pLTTCell = 0.0283
pLTOutCell = 0.0125

# Dynamic update of chemokine receptors
CXCL12crit = 60.0e-10
CXCL12recrit = 40.0e-10
CXCL13crit = 0.8e-10
CXCL13recrit = 0.6e-10

# Chemotaxis
chemoMax = 10
chemoSteep = 1e+10
chemoHalf = 2e-11
northweight = 0.1
north = np.array([0, 0, -1])

# Speed
speedCentrocyte = 7.5
speedCentroblast = 7.5
speedTCell = 10.0
speedOutCell = 3.0

# Divide and Mutate
pNow = dt * 9.0
mutationStartTime = 2.0
polarityIndex = 0.88
pDivideAgAsymmetric = 0.72
pMutAfterSelection = 0.0
pMutAffinityExponent = 1.0

# Differentiation Rates
start_differentiation = 72.0
pDif = dt * 0.1
DeleteAgInFreshCC = True
dif_delay = 6.0

pDifToOut = 0.0

# Selection Steps
testDelay = 0.02
collectFDCperiod = 0.7
antigenSaturation = 20
pSel = dt * 0.05
tcTime = 0.6
tcRescueTime = 0.5

# Antibody
pmDifferentiationRate = 24.0
antibodies_production = 1e-17
V_blood = 1e-2
N_GC = 1000
abProdFactor = antibodies_production * dt * N_GC * V_blood * 1e15
antibodyDegradation = 30
k_on = 3.6e9

expMin = 5.5
expMax = 9.5

# Movements:
Possible_Movements = list(itertools.product([-1, 0, 1], repeat=3))
Possible_Movements.remove((0, 0, 0))

# For plots/tests:
NumBCells = []
times = [0.0]


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
    print('test')
    initialise_cells()
    NumBCells.append(len(CCList) + len(CBList))
    hyphasma()
    plt.plot(times, NumBCells)
    plt.show()












    #
