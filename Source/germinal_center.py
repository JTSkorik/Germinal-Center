# Imports:
import random
import math
import numpy as np
import itertools


# Algorithm 1 (Mutation)
def mutate():
    pass


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


def update_chemokines_receptors():
    pass


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

    if random.uniform(0, 1) < prob:
        theta = random.gauss(0, 1)
        phi = random.uniform(0, 2 * math.pi)
        Polarity[ID] = turn_angle(Polarity[ID], theta, phi)
        pos = Position[ID]
        x = pos[0]
        y = pos[1]
        z = pos[2]

    if responsiveToSignalCXCL12[ID]:
        # TODO implement boundary values, could automatically be set in Grid_CXCL12.
        x_diff = Grid_CXCL12[(x + 1, y, z)] - Grid_CXCL12[(x - 1, y, z)]
        y_diff = Grid_CXCL12[(x, y + 1, z)] - Grid_CXCL12[(x, y - 1, z)]
        z_diff = Grid_CXCL12[(x, y, z + 1)] - Grid_CXCL12[(x, y, z - 1)]

        Gradient_CXCL12 = np.array([x_diff / (2 * dx), y_diff / (2 * dx), z_diff / (2 * dx)])

        mag_Gradient_CXCL12 = np.linalg.norm(Gradient_CXCL12)
        chemoFactor = (chemoMax / (
            1 + math.exp(chemoSteep * (chemoHalf - 2 * dx * mag_Gradient_CXCL12)))) * Gradient_CXCL12

        Polarity[ID] += chemoFactor

    if responsiveToSignalCXCL13[ID]:
        # TODO implement boundary values, could automatically be set in Grid_CXCL13. Given as zero in variable table.
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
                      list(itertools.product([-1, 0, 1], repeat=3)) if Movement != (0, 0, 0)]
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
def initiate_cycle(cellID, divisions):
    if divisions == 0:
        State[cell_ID] = 'cb_stop_dividing'
    else:
        State[cell_ID] = 'cb_G1'
        cycleStartTime[cellID] = 0
        endOfThisPhase[cellID] = get_duration(State[cell_ID])
        numDivisionsToDo[cellID] = divisions


def progress_cycle():
    pass


def divide_and_mutate():
    pass


# Algorithm 5 (Antigen Collection from FDCs)
def prgress_fdc_selection():
    pass


# Algorithm 6 (Screening for T cell help at the Centrocyte Stage)
def progress_tcell_selection():
    pass


# Algorithm 7 (Updating the T cells according to B cells Interactions)
def update_tcell():
    pass


def liberate_tcell():
    pass


# Algorithm 8 (Transition between Centroblasts, centrocyctes, and output Cells)
def differ_to_out():
    pass


def differ_to_cb():
    pass


def differ_to_cc():
    pass


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
        BCR[newID] = None
        '''Need BCR values'''
        Position[newID] = pos
        pMutation[newID] = p_mut(t)
        Grid_ID[pos] = newID
        Grid_Type[pos] = 'Centroblast'

        initiate_cycle(newID, numDivFounderCells)
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

        t += dt


# Extra Algorithms/Functions
def generate_spatial_points(n):
    '''Obtains a list of all points in the sphere/GC
    Input: Number of discrete diamter points, n.
    Output: List containing all points within sphere.'''

    return [(x + n / 2 + 1, y + n / 2 + 1, z + n / 2 + 1) for x in range(-n / 2, n / 2) for y in range(-n / 2, n / 2)
            for z in range(-n / 2, n / 2) if ((x + 0.5) ** 2 + (y + 0.5) ** 2 + (z + 0.5) ** 2) <= (n / 2) ** 2]


def find_empty_neighbour():
    pass


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
Polarity = {}

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

# Run Simulation:
if __name__ == "__main__":
    initialise_cells()






    #
