# Imports:
import random
import math
import numpy as np
import itertools


# Algorithm 1 (Mutation)
def mutate(ID):
    if random.uniform(0, 1) < pMutation[ID]:
        index = random.choice([0, 1, 2, 3])
        value = int(str(BCR[ID])[3 - index])
        if index == 3 and value == 1:
            BCR[ID] += 10 ** index
        else:
            if value == 0:
                BCR[ID] += 10 ** index
            elif value == 9:
                BCR[ID] -= 10 ** index
            else:
                if random.uniform(0, 1) < 0.5:
                    BCR[ID] += 10 ** index
                else:
                    BCR[ID] -= 10 ** index


# Algorithm 2 (Dynamic Updating of Chemotaxis)
def initiate_chemokine_receptors(ID, cell_type):
    if cell_type == 'Centroblast':
        responsiveToSignalCXCL12[ID] = True
        responsiveToSignalCXCL13[ID] = False
    elif cell_type == 'Centrocyte':
        responsiveToSignalCXCL12[ID] = False
        responsiveToSignalCXCL13[ID] = True
    elif cell_type == 'OutCell':
        responsiveToSignalCXCL12[ID] = False
        responsiveToSignalCXCL13[ID] = True
    else:
        print("initiate_chemokine_receptors: Invalid cell_type, {}".format(cell_type))


def update_chemokines_receptors(ID):
    pos = Position[ID]
    if Type[ID] == 'Centrocyte':
        if State[ID] == 'Unselected':
            if Grid_CXCL13[pos] > CXCL13crit:
                responsiveToSignalCXCL13[ID] = False
            elif Grid_CXCL13[pos] < CXCL13recrit:
                responsiveToSignalCXCL13[ID] = True
    elif Type[ID] == 'Centroblast':
        if Grid_CXCL12[pos] > CXCL12crit:
            responsiveToSignalCXCL12[ID] = False
        elif Grid_CXCL12[pos] < CXCL12recrit:
            responsiveToSignalCXCL12[ID] = True


# Algorithm 3 (Updating Position and Polarity of cells at each time-point)
def move(ID):
    cell_type = Type[ID]
    if cell_type == 'Centrocyte':
        prob = pLTCentrocyte
        speed = speedCentrocyte
    elif cell_type == 'Centroblast':
        prob = pLTCentroblast
        speed = speedCentroblast
    elif cell_type == 'TCell':
        prob = pLTTCell
        speed = speedTCell
    elif cell_type == 'OutCell':
        prob = pLTOutCell
        speed = speedOutCell
    else:
        prob = None
        speed = None
        print("move: Invalid cell_type, {}".format(cell_type))

    # TODO randomly assign polarity at initialisation
    if random.uniform(0, 1) < prob:
        theta = random.gauss(0, 1)
        phi = random.uniform(0, 2 * math.pi)
        Polarity[ID] = turn_angle(Polarity[ID], theta, phi)
        pos = Position[ID]
        x = pos[0]
        y = pos[1]
        z = pos[2]

    if responsiveToSignalCXCL12[ID]:
        # TODO implement boundary values, could automatically be set in Grid_CXCL12. (Have values outside sphere be constant)
        x_diff = Grid_CXCL12[(x + 1, y, z)] - Grid_CXCL12[(x - 1, y, z)]
        y_diff = Grid_CXCL12[(x, y + 1, z)] - Grid_CXCL12[(x, y - 1, z)]
        z_diff = Grid_CXCL12[(x, y, z + 1)] - Grid_CXCL12[(x, y, z - 1)]

        Gradient_CXCL12 = np.array([x_diff / (2 * dx), y_diff / (2 * dx), z_diff / (2 * dx)])

        mag_Gradient_CXCL12 = np.linalg.norm(Gradient_CXCL12)
        chemoFactor = (chemoMax / (
            1 + math.exp(chemoSteep * (chemoHalf - 2 * dx * mag_Gradient_CXCL12)))) * Gradient_CXCL12

        Polarity[ID] += chemoFactor

    if responsiveToSignalCXCL13[ID]:
        # TODO implement boundary values, could automatically be set in Grid_CXCL13. Given as zero in variable table. (Have values outside sphere be constant)
        x_diff = Grid_CXCL13[(x + 1, y, z)] - Grid_CXCL13[(x - 1, y, z)]
        y_diff = Grid_CXCL13[(x, y + 1, z)] - Grid_CXCL13[(x, y - 1, z)]
        z_diff = Grid_CXCL13[(x, y, z + 1)] - Grid_CXCL13[(x, y, z - 1)]

        Gradient_CXCL13 = np.array([x_diff / (2 * dx), y_diff / (2 * dx), z_diff / (2 * dx)])
        mag_Gradient_CXCL13 = np.linalg.norm(Gradient_CXCL13)
        chemoFactor = (chemoMax / (
            1 + math.exp(chemoSteep * (chemoHalf - 2 * dx * mag_Gradient_CXCL13)))) * Gradient_CXCL13

        Polarity[ID] += chemoFactor

    if Type[ID] == 'TCell':
        Polarity[ID] = (1.0 - northweight) * Polarity[ID] + northweight * north

    Polarity[ID] = Polarity[ID] / np.linalg.norm(Polarity[ID])
    pDifu = speed * dt / dx

    if random.uniform(0, 1) < pDifu:
        WantedPosition = np.asarray(pos) + Polarity[ID]
        Neighbours = [np.asarray(Movement) + np.asarray(pos) for Movement in
                      Possible_Movements]
        Neighbours.sort(key=lambda x: np.linalg.norm(x - WantedPosition))
        count = 0
        moved = False
        while not moved and count <= 8:
            new_pos = tuple(Neighbours[count])
            if Grid_ID[new_pos] is None:
                Position[ID] = new_pos

                Grid_ID[new_pos] = Grid_ID[pos]
                Grid_Type[new_pos] = Grid_Type[pos]
                Grid_ID[pos] = None
                Grid_Type[pos] = None

                moved = True
            count += 1


def turn_angle(pol, theta, phi):
    return 1


# Algorithm 4 (Updating events at the Centroblast Stage)
def initiate_cycle(ID):
    if numDivisionsToDo[ID] == 0:
        State[ID] = 'cb_stop_dividing'
    else:
        State[ID] = 'cb_G1'
        cycleStartTime[ID] = 0
        endOfThisPhase[ID] = get_duration(State[cell_ID])
        numDivisionsToDo[ID] = numDivisionsToDo[ID]


def progress_cycle(ID):
    cycleStartTime[ID] += dt
    if cycleStartTime[ID] > endOfThisPhase[ID]:
        # TODO restructure how cell state is stored throughout code.
        if State[ID] == 'cb_G1':
            State[ID] = 'cb_S'
        elif State[ID] == 'cb_S':
            State[ID] = 'cb_G2'
        elif State[ID] == 'cb_G2':
            State[ID] = 'cb_divide'
        if State[ID] != 'cb_divide':
            endOfThisPhase[ID] = get_duration(State[ID])
            cycleStartTime[ID] = 0


def divide_and_mutate(ID):
    global cell_ID
    pos = Position[ID]
    if random.uniform(0, 1) < pNow:
        empty_neighbours = []
        for possible_neighbour in Possible_Movements:
            new_pos = tuple(np.array(pos) + np.array(possible_neighbour))
            if Grid_ID[new_pos] is None:
                empty_neighbours.append(tuple(new_pos))
        if empty_neighbours:
            divide_pos = random.choice(empty_neighbours)

            newID = cell_ID
            cell_ID += 1

            Position[newID] = divide_pos
            Type[newID] = Type[ID]
            Grid_ID[divide_pos] = newID
            Grid_ID[divide_pos] = Type[newID]
            responsiveToSignalCXCL13[newID] = responsiveToSignalCXCL13[ID]
            responsiveToSignalCXCL12[newID] = responsiveToSignalCXCL12[ID]
            BCR[newID] = BCR[ID]
            pMutation[newID] = pMutation[ID]
            Polarity[newID] = Polarity[ID]
            numDivisionsToDo[newID] = numDivisionsToDo[ID] - 1
            numDivisionsToDo[ID] -= 1

            # TODO change initialisation to also include IAmHighAg
            IAmHighAg[ID] = False
            IAmHighAg[newID] = False

            initiate_cycle(ID)
            initiate_cycle(newID)

            if t > mutationStartTime:
                mutate(ID)
                mutate(newID)

            if random.uniform(0, 1) < pDivideAgAsymmetric:
                # TODO add retainedAg to initialisation, assume zero
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


# Algorithm 5 (Antigen Collection from FDCs)
def progress_fdc_selection(ID):
    if State[ID] == 'Unselected':
        selectedClock[ID] += dt
        if selectedClock[ID] <= collectFDCperiod:
            clock[ID] += dt
            if clock[ID] > testDelay:
                selectable[ID] = True
                # Include zero in case no neighbouring fragments, then max is zero and binding probability is zero.
                frag_antigen_amount = [0]
                for neighbour in Possible_Movements:
                    pos = Position[ID]
                    neighbour_pos = tuple(np.array(pos) + np.array(neighbour))
                    if Grid_Type[neighbour_pos] in ['Fragment', 'FCell']:
                        Frag_ID = Grid_ID[neighbour_pos]
                        frag_antigen_amount.append(FragmentAg[Frag_ID])
                pBind = affinity(ID, Antigen_Value) * max(frag_antigen_amount) / antigenSaturation
                if random.choice(0,1) < pBind:
                    State[ID] = 'Contact'
                else:
                    clock[ID] = 0
                    selectable[ID] = False
        else:
            if numFDCContacts[ID] == 0:
                State[ID] = 'Apoptosis'
            else:
                State[ID] = 'FDCSlected'

    if State[ID] == 'Contact':
        pass


# Algorithm 6 (Screening for T cell help at the Centrocyte Stage)
def progress_tcell_selection():
    pass


# Algorithm 7 (Updating the T cells according to B cells Interactions)
def update_tcell():
    pass


def liberate_tcell():
    pass


# Algorithm 8 (Transition between Centroblasts, Centrocyctes, and Output Cells)
def differ_to_out(ID):
    OutList.append(ID)
    initiate_chemokine_receptors(ID, 'OutCell')
    # TODO NumOutCells
    Grid_Type[Position[ID]] = 'OutCell'


def differ_to_cb(ID):
    CBList.append(ID)
    initiate_chemokine_receptors(ID, 'Centroblast')
    Grid_Type[Position[ID]] = 'Centroblast'
    # Unfinished


def differ_to_cc(ID):
    CCList.append(ID)
    initiate_chemokine_receptors(ID, 'Centrocyte')
    Grid_Type[Position[ID]] = 'Centrocyte'
    State[ID] = 'Unselected'
    if DeleteAgInFreshCC:
        numFDCContacts[ID] = 0
    else:
        numFDCContacts[ID] = retainedAg[ID] + 0.5


# Algorithm 9 (Initialisation)
def initialise_cells():
    global cell_ID

    # Initialise Stromal Cells:
    for _ in range(NumStromalCells):
        # Find empty location in dark zone
        pos = random.choice(DarkZone)
        while Grid_ID[pos] is not None:
            pos = random.choice(DarkZone)

        # Obtain new ID for new cell
        newID = cell_ID
        cell_ID += 1

        # Add to appropriate lists and dictionaries
        StormaList.append(newID)
        Type[newID] = 'Stromal'
        Position[newID] = pos
        Grid_ID[pos] = newID
        Grid_Type[pos] = 'Stromal'

    # Initialise Fragments:
    for _ in range(NumFDC):
        # Find empty location in light zone
        pos = random.choice(LightZone)
        while Grid_ID[pos] is not None:
            pos = random.choice(LightZone)

        # Obtain new ID for new cell
        newID = cell_ID
        cell_ID += 1

        # Add to appropriate lists and dictonaries
        FDCList.append(newID)
        Type[newID] = 'FCell'
        Position[newID] = pos
        Grid_ID[pos] = newID
        Grid_Type[pos] = 'FCell'

        # Find the fragments for the FCell
        FCell_ID = cell_ID
        fragments = []
        x = pos[0]
        y = pos[1]
        z = pos[2]
        for i in range(1, DendriteLength + 1):
            # TODO implement O(1) checks for whether position is valid.
            for frag_pos in [(x + i, y, z), (x - i, y, z), (x, y + i, z), (x, y - i, z)]:
                if frag_pos in LightZone and Grid_ID[frag_pos] is None:
                    newID = cell_ID
                    cell_ID += 1
                    fragments.append(newID)
                    Position[newID] = pos
                    Grid_ID[frag_pos] = newID
                    Grid_Type[frag_pos] = 'Fragment'

            # When Z axis is changing, we require extra check that we're still in light zone.
            for frag_pos in [(x, y, z + i), (x, y, z - i)]:
                if frag_pos in LightZone and Grid_ID[frag_pos] is None:
                    newID = cell_ID
                    cell_ID += 1
                    fragments.append(newID)
                    Position[newID] = pos
                    Grid_ID[frag_pos] = newID
                    Grid_Type[frag_pos] = 'Fragment'

        Fragments[FCell_ID] = fragments

        # Assign each fragment an amount of antigen
        FCellVol[FCell_ID] = len(fragments) + 1  # +1 accounts for centre
        agPerFrag = AntigenAmountPerFDC / FCellVol[FCell_ID]
        for Frag in [FCell_ID] + Fragments[FCell_ID]:
            FragmentAg[Frag] = agPerFrag

    # Initialise Seeder Cells:
    for _ in range(NumSeeder):
        pos = random.choice(LightZone)  # Find empty location in list zone
        while Grid_ID[pos] is not None:
            pos = random.choice(LightZone)

        # Obtain new ID for new cell
        newID = cell_ID
        cell_ID += 1

        # Add cell to appropriate lists and dictionaries
        CBList.append(newID)
        Type[newID] = 'Centroblast'
        # TODO make up BCR values
        BCR[newID] = None
        Position[newID] = pos
        pMutation[newID] = p_mut(t)
        numDivisionsToDo[newID] = numDivFounderCells
        Grid_ID[pos] = newID
        Grid_Type[pos] = 'Centroblast'

        initiate_cycle(newID)
        initiate_chemokine_receptors(newID, 'Centroblast')

    # Initialise T Cells:
    for _ in range(NumTC):
        pos = random.choice(LightZone)  # Find empty location in light zone
        while Grid_ID[pos] is not None:
            pos = random.choice(LightZone)

        # Obtain new ID for new cell.
        newID = cell_ID
        cell_ID += 1

        # Add cell to appropriate lists and dictionaries
        TCList.append(newID)
        Type[newID] = 'TCell'
        Position[newID] = pos
        Grid_ID[pos] = newID
        Grid_Type[pos] = 'TCell'


# Algorithm 10 (Hyphasma: Simulation of Germinal Center)
def hyphasma():
    global t
    global cell_ID
    while t <= tmax:
        for StromalCell in StormaList:
            signal_secretion(StromalCell, 'CXCL12', p_mkCXCL12)
        for FCell in random.shuffle(FDCList):
            signal_secretion(FCell, 'CXCL13', p_mkCXCL13)
            fragments = Fragments[FCell]
            for frag in fragments:
                pass
                # TODO lines 9 to 12

        diffuse_signal('CXCL12', 'CXCL13')
        # TODO lines 17 to 24

        for ID in random.shuffle(OutList):
            move(ID)
            pos = Position[ID]
            if is_surface_point(pos):
                OutList.remove(ID)

        for ID in random.shuffle(CBList):
            update_chemokines_receptors(ID)
            progress_cycle(ID)
            if State[ID] == 'cb_divide':
                divide_and_mutate(ID)
            if State[ID] == 'cb_stop_diving':
                if random.uniform(0, 1) < pDif:
                    if IAmHighAg[ID]:
                        differ_to_out(ID)
                    else:
                        differ_to_cc(ID)

            if State[ID] != 'cb_M':
                move(ID)

        for ID in random.shuffle(CCList):
            update_chemokines_receptors(ID)
            progress_fdc_selection(ID)

        t += dt


# Extra Algorithms/Functions
def generate_spatial_points(n):
    '''Obtains a list of all points in the sphere/GC
    Input: Number of discrete diamter points, n.
    Output: List containing all points within sphere.'''

    return [(x + n / 2 + 1, y + n / 2 + 1, z + n / 2 + 1) for x in range(-n / 2, n / 2) for y in range(-n / 2, n / 2)
            for z in range(-n / 2, n / 2) if ((x + 0.5) ** 2 + (y + 0.5) ** 2 + (z + 0.5) ** 2) <= (n / 2) ** 2]


def affinity(ID, antigen):
    return 1


def p_mut(time):
    '''Finds the probability of mutation
    Input: Current time, t
    Output: Probability of Mutation'''

    if time > 24:
        return 0.5
    else:
        return 0


def get_duration(cell_state):
    '''Finds duration before cell divides using Guassian random variable.
    Input: current state of cell, cell_state
    Output: amount of time
    '''
    sigma = 1
    if cell_state == 'cb_G1':
        mu = 2.0
    elif cell_state == 'cb_S':
        mu = 1.0
    elif cell_state == 'cb_G2':
        mu = 2.5
    elif cell_state == 'cb_M':
        mu = 0.5
    else:
        mu = None
        print("getDuration: Invalid cell state, {}".format(cell_state))

    return random.gauss(mu, sigma)


def signal_secretion(ID, chem, chem_prod_rate):
    pass


def diffuse_signal(Chem1, Chem2):
    pass


def is_surface_point(position):
    pass


# Set-up for simulation:
Antigen_Value = 1234

# Distance Variables:
N = 64  # Diameter of sphere/GC
AllPoints = generate_spatial_points(N)
DarkZone = [point for point in AllPoints if point[2] > N / 2]
LightZone = [point for point in AllPoints if point[2] <= N / 2]

dx = 5

# Time Variables:
dt = 0.002
tmin = 0.0
tmax = 504.0
t = 0.0  # Current time

# Initialisation
NumStromalCells = 300
NumFDC = 200
NumSeeder = 3
NumTC = 250

DendriteLength = 8
AntigenAmountPerFDC = 3000

# Lists to store ID of each cell in each state (and fragments)
StormaList = []
FDCList = []
CBList = []
CCList = []
TCList = []
OutList = []

# Here we will create empty dictionaries to store different properties. Will add them as necessary.
BCR = {}
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

# Dictionaries storing what is at each location. Initially empty, so 'None'.
Grid_ID = {pos: None for pos in AllPoints}
Grid_Type = {pos: None for pos in AllPoints}

# Dictionaries storing amounts of CXCL12 and CXCL13 at each point:
Grid_CXCL12 = {pos: None for pos in AllPoints}
Grid_CXCL13 = {pos: None for pos in AllPoints}

# Sequence variable for giving each cell an ID:
cell_ID = 0

# Dynamic number of divisions:
numDivFounderCells = 12

# Production/ Diffusion Rates:
p_mkCXCL12 = 4e-7
p_mkCXCL13 = 1e-8

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
mutationStartTime = 24.0
polarityIndex = 0.88
pDivideAgAsymmetric = 0.72

# Differentiation Rates
pDif = dt * 0.1
DeleteAgInFreshCC = True

# Selection Steps
testDelay = 0.02
collectFDCperiod = 0.7
antigenSaturation = 20

# Movements:
Possible_Movements = list(itertools.product([-1, 0, 1], repeat=3))
Possible_Movements.remove((0, 0, 0))

# Run Simulation:
if __name__ == "__main__":
    initialise_cells()






    #
