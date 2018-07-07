# Imports:
import random
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt
from enum import Enum


# TODO CXCL12 and CXCL13 made up values.
# TODO documentation and comments
# TODO signal_secretion, diffusion, and turn_angle functions.


def mutate(ID):
    """
    Algorithm 1, Mutation.
    Mutates the BCR value for a cell with a certain probability.
    :param ID: integer, determines which cell in the pop we are talking about.
    :return:
    """
    # Determine where the BCR value should mutate.
    if random.uniform(0, 1) < pMutation[ID]:
        # Choose index and value of BCR to change.
        index = random.choice([0, 1, 2, 3])
        value = int(str(Cell_BCR[ID])[3 - index])
        # Randomly apply plus or minus one to one BCR position.
        if index == 3 and value == 1:
            Cell_BCR[ID] += 10 ** index
        else:
            if value == 0:
                Cell_BCR[ID] += 10 ** index
            elif value == 9:
                Cell_BCR[ID] -= 10 ** index
            else:
                if random.uniform(0, 1) < 0.5:
                    Cell_BCR[ID] += 10 ** index
                else:
                    Cell_BCR[ID] -= 10 ** index
    # If new BCR value obtained, we need to start tracking its stats.
    if Cell_BCR[ID] not in BCR_values_all:
        BCR_values_all.add(Cell_BCR[ID])
        NumBCROutCells[Cell_BCR[ID]] = 0
        NumBCROutCellsProduce[Cell_BCR[ID]] = 0
        AntibodyPerBCR[Cell_BCR[ID]] = 0


def initiate_chemokine_receptors(ID):
    """
    Algorithm 2, Dynamic Updating of Chemotaxis.
    Used to determine the attributes responsive_to_CXCL12 and responsive_to_CXCL13 of a cell given its type.
    :param ID: integer, determines which cell in the pop we are talking about.
    :return:
    """
    if Type[ID] == CellType.Centroblast:
        responsiveToSignalCXCL12[ID] = True
        responsiveToSignalCXCL13[ID] = False
    elif Type[ID] == CellType.Centrocyte:
        responsiveToSignalCXCL12[ID] = False
        responsiveToSignalCXCL13[ID] = True
    elif Type[ID] == CellType.OutCell:
        responsiveToSignalCXCL12[ID] = False
        responsiveToSignalCXCL13[ID] = True
    else:
        print("initiate_chemokine_receptors: Invalid cell_type, {}".format(Type[ID]))


def update_chemokines_receptors(ID):
    """
    Algorithm 2, Dynamic Updating of Chemotaxis.
    Used to update the attributes responsive_to_CXCL12 and responsive_to_CXCL13 of a cell given its type and state.
    :param ID: integer, determines which cell in the pop we are talking about.
    :return:
    """
    pos = Position[ID]
    if Type[ID] == CellType.Centrocyte:
        if State[ID] == CellState.unselected:
            if Grid_CXCL13[pos] > CXCL13crit:
                responsiveToSignalCXCL13[ID] = False
            elif Grid_CXCL13[pos] < CXCL13recrit:
                responsiveToSignalCXCL13[ID] = True
    elif Type[ID] == CellType.Centroblast:
        if Grid_CXCL12[pos] > CXCL12crit:
            responsiveToSignalCXCL12[ID] = False
        elif Grid_CXCL12[pos] < CXCL12recrit:
            responsiveToSignalCXCL12[ID] = True


def move(ID):
    # TODO add check that the cells are staying inside their respective zones.
    """
    Algorithm 3, Updating Position and Polarity of cells at each time-point.
    Updates the polarity of a cell and then will move the cell within the Germinal center. Both events occur stochastically.
    :param ID: integer, determines which cell in the pop we are talking about.
    :return:
    """

    # Obtain current position of cell.
    pos = Position[ID]
    x = pos[0]
    y = pos[1]
    z = pos[2]

    # Obtain required parameters
    cell_type = Type[ID]
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
        if responsiveToSignalCXCL12[ID]:
            x_diff = Grid_CXCL12[(x + 1, y, z)] - Grid_CXCL12[(x - 1, y, z)]
            y_diff = Grid_CXCL12[(x, y + 1, z)] - Grid_CXCL12[(x, y - 1, z)]
            z_diff = Grid_CXCL12[(x, y, z + 1)] - Grid_CXCL12[(x, y, z - 1)]

            Gradient_CXCL12 = np.array([x_diff / (2 * dx), y_diff / (2 * dx), z_diff / (2 * dx)])

            mag_Gradient_CXCL12 = np.linalg.norm(Gradient_CXCL12)
            chemoFactor = (chemoMax / (
                1 + math.exp(chemoSteep * (chemoHalf - 2 * dx * mag_Gradient_CXCL12)))) * Gradient_CXCL12

            Polarity[ID] += chemoFactor

        # Influence due to CXCL13
        if responsiveToSignalCXCL13[ID]:
            x_diff = Grid_CXCL13[(x + 1, y, z)] - Grid_CXCL13[(x - 1, y, z)]
            y_diff = Grid_CXCL13[(x, y + 1, z)] - Grid_CXCL13[(x, y - 1, z)]
            z_diff = Grid_CXCL13[(x, y, z + 1)] - Grid_CXCL13[(x, y, z - 1)]

            Gradient_CXCL13 = np.array([x_diff / (2 * dx), y_diff / (2 * dx), z_diff / (2 * dx)])
            mag_Gradient_CXCL13 = np.linalg.norm(Gradient_CXCL13)
            chemoFactor = (chemoMax / (
                1 + math.exp(chemoSteep * (chemoHalf - 2 * dx * mag_Gradient_CXCL13)))) * Gradient_CXCL13

            Polarity[ID] += chemoFactor

        # Influence specific to T Cells
        if cell_type == CellType.TCell:
            Polarity[ID] = (1.0 - northweight) * Polarity[ID] + northweight * north

        Polarity[ID] = Polarity[ID] / np.linalg.norm(Polarity[ID])

    # Probability of movement
    pDifu = speed * dt / dx

    if random.uniform(0, 1) < pDifu:
        # Find possible new positions based in order of best preference
        WantedPosition = np.asarray(pos) + Polarity[ID]
        Neighbours = [np.asarray(Movement) + np.asarray(pos) for Movement in
                      Possible_Movements if np.linalg.norm(
                np.asarray(Movement) + np.asarray(pos) - np.array(OffSet)) <= (N / 2)]
        Neighbours.sort(key=lambda x: np.linalg.norm(x - WantedPosition))

        # Move the cell to best available position that isn't against direction of polarity
        count = 0
        moved = False
        while not moved and count <= 9:
            new_pos = tuple(Neighbours[count])
            if Grid_ID[new_pos] is None:
                Position[ID] = new_pos

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
    v = np.random.standard_normal(3)
    Polarity[ID] = v / np.linalg.norm(v)


def initiate_cycle(ID):
    """
    Algorithm 3, Updating events at the Centroblast Stage.
    Sets the state for a cell depending on how many divisions left to complete.
    :param ID: integer, determines which cell in the pop we are talking about.
    :return:
    """
    if numDivisionsToDo[ID] == 0:
        State[ID] = CellState.Stop_Dividing
    else:
        State[ID] = CellState.cb_G1
        cycleStartTime[ID] = 0
        endOfThisPhase[ID] = get_duration(ID)
        numDivisionsToDo[ID] = numDivisionsToDo[ID]


def progress_cycle(ID):
    """
    Algorithm 4, Updating events at the Centroblast Stage.
    Progresses the cell to its next state and calculates how long it till next change of state.
    :param ID: integer, determines which cell in the pop we are talking about.
    :return:
    """
    # Progress cell into its next state
    cycleStartTime[ID] += dt
    if cycleStartTime[ID] > endOfThisPhase[ID]:
        if State[ID] == CellState.cb_G1:
            State[ID] = CellState.cb_S
        elif State[ID] == CellState.cb_S:
            State[ID] = CellState.cb_G2
        elif State[ID] == CellState.cb_G2:
            State[ID] = CellState.cb_divide

        # Finds time till end of new state.
        if State[ID] != CellState.cb_divide:
            endOfThisPhase[ID] = get_duration(ID)
            cycleStartTime[ID] = 0


def divide_and_mutate(ID, t):
    """
    Algorithm 4, Updating events at the Centroblast Stage.
    Takes a cell and divides it with probability pNow. Will also attempt to mutate the cells.
    :param ID: integer, determines which cell in the pop we are talking about.
    :param t: float, current time of the simulation.
    :return:
    """
    if random.uniform(0, 1) < pNow:
        # Find all empty positions neighbouring cell.
        pos = Position[ID]
        empty_neighbours = [tuple(np.array(pos) + np.array(possible_neighbour)) for possible_neighbour in
                            Possible_Movements if np.linalg.norm(
                            np.asarray(possible_neighbour) + np.asarray(pos) - np.array(OffSet)) <= (N / 2)]

        # Randomly choose one position for new cell
        if empty_neighbours:
            divide_pos = random.choice(empty_neighbours)

            # Generate a new ID for the cell and copy over the properties from the old cell.
            newID = AvailableCellIDs.pop()
            Position[newID] = divide_pos
            Type[newID] = Type[ID]
            Grid_ID[divide_pos] = newID
            Grid_ID[divide_pos] = Type[newID]
            responsiveToSignalCXCL13[newID] = responsiveToSignalCXCL13[ID]
            responsiveToSignalCXCL12[newID] = responsiveToSignalCXCL12[ID]
            Cell_BCR[newID] = Cell_BCR[ID]
            pMutation[newID] = pMutation[ID]
            Polarity[newID] = Polarity[ID]
            numDivisionsToDo[newID] = numDivisionsToDo[ID] - 1
            numDivisionsToDo[ID] -= 1

            IAmHighAg[ID] = False
            IAmHighAg[newID] = False

            # Initiate the cycle for each cell.
            initiate_cycle(ID)
            initiate_cycle(newID)

            # Mutate the cells
            if t > mutationStartTime:
                mutate(ID)
                mutate(newID)

            # Assign amount of retained antigen to each cell
            if random.uniform(0, 1) < pDivideAgAsymmetric:
                if retainedAg[ID] == 0:
                    retainedAg[newID] = 0
                else:
                    sep = random.gauss(polarityIndex, 1)
                    while sep < 0 or sep > 1:
                        sep = random.gauss(polarityIndex, 1)

                    retainedAg[newID] = sep * retainedAg[ID]
                    retainedAg[ID] = (1 - sep) * retainedAg[ID]

                    if sep > 0.5:
                        IAmHighAg[newID] = True
                    else:
                        IAmHighAg[ID] = False
            else:
                retainedAg[newID] = retainedAg[ID] / 2
                retainedAg[ID] = retainedAg[ID] / 2


def progress_fdc_selection(ID):
    """
    Algorithm 5, Antigen Collection from FDCs.
    Allows for centrocytes to collect antigen from neighbouring F cells / fragments.
    :param ID: integer, determines which cell in the pop we are talking about.
    :return:
    """

    if State[ID] == CellState.Unselected:
        # Progress selected clock and check if able to collect antigen.
        selectedClock[ID] += dt
        if selectedClock[ID] <= collectFDCperiod:
            clock[ID] += dt
            if clock[ID] > testDelay:
                selectable[ID] = True
                # Find frag component with largest amount of antigen.
                Frag_max = None
                Frag_max_ID = None
                pos = Position[ID]
                for neighbour in Possible_Movements:
                    neighbour_pos = tuple(np.array(pos) + np.array(neighbour))
                    if Grid_Type[neighbour_pos] in [CellType.Fragment, CellType.FCell]:
                        Frag_ID = Grid_ID[neighbour_pos]
                        if FragmentAg[ID] > Frag_max:
                            Frag_max = FragmentAg[ID]
                            Frag_max_ID = Frag_ID
                pBind = affinity(ID) * Frag_max / antigenSaturation
                # Bind cell and fragment with probability pBind or reset clock.
                if random.choice(0, 1) < pBind:
                    State[ID] = CellState.FDCcontact
                    Frag_Contacts[ID] = Frag_max_ID
                else:
                    clock[ID] = 0
                    selectable[ID] = False
        else:
            # Cell dies if it doesn't get any contacts.
            if numFDCContacts[ID] == 0:
                State[ID] = CellState.Apoptosis
            else:
                State[ID] = CellState.FDCselected

    elif State[ID] == CellState.Contact:
        selectedClock[ID] += dt
        if random.uniform(0, 1) < pSel:
            numFDCContacts[ID] += 1
            Frag_ID = Frag_Contacts[ID]
            FragmentAg[Frag_ID] -= 1
            State[ID] = CellState.Unselected
            clock[ID] = 0
            selectable[ID] = False


def progress_tcell_selection(ID, t):
    """
    Algorithm 6, Screening for T cell help at Centrocyte Stage.

    :param ID: integer, determines which cell in the pop we are talking about.
    :param t: t: float, current time of the simulation.
    :return:
    """
    if State[ID] == CellState.FDCselected:
        # Find if there is a neighbouring T cell.
        pos = Position[ID]
        for neighbour in Possible_Movements:
            neighbour_pos = tuple(np.array(pos) + np.array(neighbour))
            # If there is a neighbouring T cell, we record the contact.
            if Grid_Type[neighbour_pos] == CellType.TCell and State[ID] != CellState.TCcontact:
                update_tcell(ID, Grid_ID[neighbour_pos])
                State[ID] = CellState.TCcontact
                tcClock[ID] = 0
                tcSignalDuration[ID] = 0

    elif State[ID] == CellState.TCcontact:
        tcClock[ID] += dt
        # Check is current cell has least amount of antigens compared to T cells neighbouring cells.
        lowest_antigen = True
        for ID_BC in TCellInteractions[BCellInteractions[ID]]:
            if ID != ID_BC and retainedAg[ID] <= retainedAg[ID_BC]:
                lowest_antigen = False
        if lowest_antigen:
            tcSignalDuration[ID] += dt
        if tcSignalDuration[ID] > tcRescueTime:
            State[ID] = CellState.Selected
            selectedClock[ID] = 0
            rand = random.uniform(0, 1)
            IndividualDifDelay[ID] = dif_delay * (1 + 0.1 * math.log(1 - rand) / rand)
            liberate_tcell(ID, BCellInteractions[ID])
        elif tcClock[ID] >= tcTime:
            State[ID] = CellState.Apoptosis
            liberate_tcell(ID, BCellInteractions[ID])

    elif State[ID] == CellState.Selected:
        selectedClock[ID] += dt
        if selectedClock[ID] > IndividualDifDelay[ID]:
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
    TCellInteractions[ID_T].append(ID_B)
    BCellInteractions[ID_B] = ID_T
    State[ID_T] = CellState.TC_CC_Contact


def liberate_tcell(ID_B, ID_T):
    """
    Algorithm 7, Updating the T cells according to B cells interactions.
    Updates records of whether cells are interacting, specifically removing cells from being interactive.
    :param ID_B: integer, determines which B cell in the pop we are talking about.
    :param ID_T: integer, determines which T cell in the pop we are talking about.
    :return:
    """
    TCellInteractions[ID_T].remove(ID_B)
    BCellInteractions[ID_B] = None
    if not TCellInteractions[ID_T]:
        State[ID_T] = CellState.TCnormal


def differ_to_out(ID):
    """
    Algorithm 8, Transition between Centroblasts, Centrocyctes, and Output Cells.
    Transition a cell from Centroblast or Centrocycte to an Output Cell.
    :param ID: integer, determines which cell in the pop we are talking about.
    :return:
    """
    # Update cell and Grid position properties.
    OutList.append(ID)
    Type[ID] = CellType.OutCell
    Grid_Type[Position[ID]] = CellType.OutCell
    State[ID] = CellState.OutCell
    initiate_chemokine_receptors(ID)
    NumBCROutCells[Cell_BCR[ID]] += 1


def differ_to_cb(ID, t):
    """
    Algorithm 8, Transition between Centroblasts, Centrocyctes, and Output Cells.
    Transitions a cell from Centrocyte to Centroblast.
    :param ID: integer, determines which cell in the pop we are talking about.
    :param t: float,  current time of the simulation.
    :return:
    """
    CBList.append(ID)
    Type[ID] = CellType.Centroblast
    initiate_chemokine_receptors(ID)
    Grid_Type[Position[ID]] = CellType.Centroblast

    # Find new probability of mutation.
    pMutation[ID] = p_mut(t) + (pMutAfterSelection - p_mut(t)) * affinity(ID) ** pMutAffinityExponent
    agFactor = numFDCContacts[ID] ** pMHCdepHill
    num_div = pMHCdepMin + (pMHCdepMax - pMHCdepMin) * agFactor / (agFactor + pMHCdepK ** pMHCdepHill)
    numDivisionsToDo[ID] = num_div
    initiate_cycle(ID)

    retainedAg[ID] = numFDCContacts[ID]
    IAmHighAg[ID] = True


def differ_to_cc(ID):
    """
    Algorithm 8, Transition between Centroblasts, Centrocyctes, and Output Cells.
    Transitions a cell from Centroblast to Centrocyte.
    :param ID: integer, determines which cell in the pop we are talking about.
    :return:
    """
    CCList.append(ID)
    Type[ID] = CellType.Centrocyte
    initiate_chemokine_receptors(ID)
    Grid_Type[Position[ID]] = CellType.Centrocyte
    State[ID] = CellState.Unselected
    if DeleteAgInFreshCC:
        numFDCContacts[ID] = 0
    else:
        numFDCContacts[ID] = retainedAg[ID] + 0.5


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
        Type[ID] = CellType.Stromal
        Position[ID] = pos
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
        Type[ID] = CellType.FCell
        Position[ID] = pos
        Grid_ID[pos] = ID
        Grid_Type[pos] = CellType.FCell

        # Find the fragments for the FCell
        FCell_ID = ID
        fragments = []
        x = pos[0]
        y = pos[1]
        z = pos[2]
        for i in range(1, DendriteLength + 1):
            for frag_pos in [(x + i, y, z), (x - i, y, z), (x, y + i, z), (x, y - i, z), (x, y, z - i)]:
                if (frag_pos[0] - (N / 2 + 0.5)) ** 2 + (frag_pos[1] - (N / 2 + 0.5)) ** 2 + (
                            frag_pos[2] - (N / 2 + 0.5)) ** 2 <= (N / 2) ** 2 and Grid_ID[frag_pos] is None:
                    ID = AvailableCellIDs.pop()
                    fragments.append(ID)
                    Position[ID] = pos
                    Grid_ID[frag_pos] = ID
                    Grid_Type[frag_pos] = CellType.Fragment

            # When Z axis is increasing, we require an extra check to ensure that we're still in light zone.
            frag_pos = (x, y, z + i)
            if (frag_pos[0] - (N / 2 + 0.5)) ** 2 + (frag_pos[1] - (N / 2 + 0.5)) ** 2 + (
                        frag_pos[2] - (N / 2 + 0.5)) ** 2 <= (N / 2) ** 2 and frag_pos[2] <= N / 2 and Grid_ID[
                frag_pos] is None:
                ID = AvailableCellIDs.pop()
                fragments.append(ID)
                Position[ID] = pos
                Grid_ID[frag_pos] = ID
                Grid_Type[frag_pos] = CellType.Fragment

        Fragments[FCell_ID] = fragments

        # Assign each fragment an amount of antigen
        FCellVol[FCell_ID] = len(fragments) + 1  # +1 accounts for centre
        agPerFrag = AntigenAmountPerFDC / FCellVol[FCell_ID]
        for Frag in [FCell_ID] + Fragments[FCell_ID]:
            FragmentAg[Frag] = agPerFrag
            icAmount[Frag] = 0

    # Initialise Seeder Cells:
    for _ in range(NumSeeder):
        pos = random.choice(LightZone)  # Find empty location in list zone
        while Grid_ID[pos] is not None:
            pos = random.choice(LightZone)

        # Obtain new ID for new cell
        ID = AvailableCellIDs.pop()

        # Add cell to appropriate lists and dictionaries
        CBList.append(ID)
        Type[ID] = CellType.Centroblast
        Cell_BCR[ID] = random.choice(tuple(BCR_values_all))
        Position[ID] = pos
        pMutation[ID] = p_mut(0.0)
        numDivisionsToDo[ID] = numDivFounderCells
        Grid_ID[pos] = ID
        Grid_Type[pos] = CellType.Centroblast
        v = np.random.standard_normal(3)
        Polarity[ID] = v / np.linalg.norm(v)
        IAmHighAg[ID] = False
        retainedAg[ID] = 0

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
        Type[ID] = CellType.TCell
        Position[ID] = pos
        Grid_ID[pos] = ID
        Grid_Type[pos] = CellType.TCell
        v = np.random.standard_normal(3)
        Polarity[ID] = v / np.linalg.norm(v)
        TCellInteractions[ID] = []
        State[ID] = CellState.TCnormal
        responsiveToSignalCXCL12[ID] = False
        responsiveToSignalCXCL13[ID] = False


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
        for StromalCell in StormaList:
            signal_secretion(StromalCell, 'CXCL12', p_mkCXCL12)

        # Randomly iterate over F cells / Fragments.
        random.shuffle(FDCList)
        for FCell in FDCList:
            # Secrete CXCL13 from
            signal_secretion(FCell, 'CXCL13', p_mkCXCL13)
            fragments = Fragments[FCell]
            # Update antigen amounts for each fragment.
            for frag in fragments:
                for bcr_seq in BCR_values_all:
                    d_ic = dt * (
                        k_on * FragmentAg[frag] * AntibodyPerBCR[bcr_seq] - k_off(bcr_seq) * icAmount[frag])
                    FragmentAg[frag] -= d_ic
                    icAmount[frag] += d_ic

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
            # Move cell and remove if on surface of sphere / Germinal Center.
            move(ID)
            pos = Position[ID]
            if is_surface_point(pos):
                OutList.remove(ID)
                AvailableCellIDs.append(ID)

        # Randomly iterate over Centroblast cells.
        random.shuffle(CBList)
        for ID in CBList:
            # Progress cells in their lifetime cycle.
            update_chemokines_receptors(ID)
            progress_cycle(ID)
            # Attempt to divide cell if ready.
            if State[ID] == CellState.cb_divide:
                divide_and_mutate(ID, t)
            if State[ID] == CellState.Stop_Dividing:
                if random.uniform(0, 1) < pDif:
                    if IAmHighAg[ID]:
                        differ_to_out(ID)
                    else:
                        differ_to_cc(ID)
            # Move allowed cells.
            if State[ID] != CellState.cb_M:
                move(ID)
        # Randomly iterate over Centrocyte cells.
        random.shuffle(CCList)
        for ID in CCList:
            # Update cells progress
            update_chemokines_receptors(ID)
            progress_fdc_selection(ID)
            progress_tcell_selection(ID, t)
            # Remove cell from simulation if dead or move if not in contact with T or F cell.
            if State[ID] == CellState.Apoptosis:
                CBList.remove(ID)
                AvailableCellIDs.append(ID)
            elif State[ID] not in [CellState.FDCcontact, CellState.TCcontact]:
                move(ID)
        # Randomly iterate over T cells and move if not attached to another cell.
        random.shuffle(TCList)
        for ID in TCList:
            if State[ID] == CellState.TCnormal:
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
    return [(x + n / 2 + 1, y + n / 2 + 1, z + n / 2 + 1) for x in range(-n / 2, n / 2) for y in range(-n / 2, n / 2)
            for z in range(-n / 2, n / 2) if ((x + 0.5) ** 2 + (y + 0.5) ** 2 + (z + 0.5) ** 2) <= (n / 2) ** 2]


def affinity(ID):
    """
    Calculates the affinity between the Antigen and a given cells BCR.
    :param ID: integer, determines which cell in the pop we are talking about.
    :return: float, affinity value.
    """
    hamming_dist = sum(el1 != el2 for el1, el2 in zip(str(Cell_BCR[ID]), str(Antigen_Value)))
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


def get_duration(ID):
    """
    Find duration of time before a cell divides. Amount of time determined using Guassian Random Variable.
    :param ID: integer, determines which cell in the pop we are talking about.
    :return: float, sample from a Guassian random variable representing time till a cell divides.
    """
    sigma = 1
    if State[ID] == CellState.cb_G1:
        mu = 2.0
    elif State[ID] == CellState.cb_S:
        mu = 1.0
    elif State[ID] == CellState.cb_G2:
        mu = 2.5
    elif State[ID] == CellState.cb_M:
        mu = 0.5
    else:
        mu = None
        print("getDuration: Invalid cell state, {}".format(State[ID]))

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
DarkZone = [point for point in AllPoints if point[2] > N / 2]
LightZone = [point for point in AllPoints if point[2] <= N / 2]
OffSet = (N / 2 + 0.5, N / 2 + 0.5, N / 2 + 0.5)

# Available Cell IDs:
AvailableCellIDs = list(range(len(AllPoints)))

# Spatial step size (micrometers)
dx = 5

# Time Variables:
dt = 0.002
tmin = 0.0
tmax = 20.0

# Initialisation
NumStromalCells = 30
NumFDC = 20
NumSeeder = 1
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

# Here we will create empty dictionaries to store different properties. Will add them as necessary.
Cell_BCR = {}
pMutation = {}
Type = {}
State = {}
Position = {}
cycleStartTime = {}
endOfThisPhase = {}
numDivisionsToDo = {}
responsiveToSignalCXCL12 = {}
responsiveToSignalCXCL13 = {}
Fragments = {}
FragmentAg = {}
FCellVol = {}
numFDCContacts = {}
Polarity = {}
IAmHighAg = {}
retainedAg = {}
selectedClock = {}
selectable = {}
clock = {}
Frag_Contacts = {}
tcClock = {}
tcSignalDuration = {}
IndividualDifDelay = {}
icAmount = {}

# Dictionaries storing what is at each location. Initially empty, so 'None'.
Grid_ID = {pos: None for pos in AllPoints}
Grid_Type = {pos: None for pos in AllPoints}

# Dictionaries storing amounts of CXCL12 and CXCL13 at each point:
Grid_CXCL12 = {pos: random.uniform(80e-11, 80e-10) for pos in
               [(x + N / 2 + 1, y + N / 2 + 1, z + N / 2 + 1) for x in range(-N / 2, N / 2) for y in
                range(-N / 2, N / 2)
                for z in range(-N / 2, N / 2)]}
Grid_CXCL13 = {pos: random.uniform(0.1e-10, 0.1e-9) for pos in
               [(x + N / 2 + 1, y + N / 2 + 1, z + N / 2 + 1) for x in range(-N / 2, N / 2) for y in
                range(-N / 2, N / 2)
                for z in range(-N / 2, N / 2)]}

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
mutationStartTime = 5.0
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
    initialise_cells()
    NumBCells.append(len(CCList) + len(CBList))
    hyphasma()
    plt.plot(times, NumBCells)
    plt.show()












    #
