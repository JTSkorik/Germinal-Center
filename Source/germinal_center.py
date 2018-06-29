# Imports:
import random
from enum import Enum


class CellType(Enum):
    CENTROBLAST = 1
    CENTROCYTE = 2
    OUTCELL = 3




def mutate():
    """
    Algorithm 1 To be writtern
    :return:
    """
    pass


def initiate_chemokine_receptors(cellID, cell_type):
    """
    Algorithm 2 (Dynamic Updating of Chemotaxis)
    :param cellID:  integer, determines which cell in the pop we are talking about
    :param cell_type: str, determines the type of cell
    :return:
    """
    if cell_type == CellType.CENTROBLAST:
        responsiveToSignalCXCL12[cellID] = True
        responsiveToSignalCXCL13[cellID] = False
    elif cell_type == CellType.CENTROCYTE:
        responsiveToSignalCXCL12[cellID] = False
        responsiveToSignalCXCL13[cellID] = True
    elif cell_type == CellType.OUTCELL:
        responsiveToSignalCXCL12[cellID] = False
        responsiveToSignalCXCL13[cellID] = True
    else:
        print("initiateChemokineReceptors: Invalid cell_type, {}".format(cell_type))


def update_chemokines_receptors():
    pass


# Algorithm 3 (Updating Position and Polarity of cells at each time-point)
def move():
    pass


def turn_angle():
    pass


# Algorithm 4 (Updating events at the CENTROBLAST Stage)
def initiate_cycle(cellID, divisions):
    if divisions == 0:
        State[cell_ID] = 'cb_stop_dividing'
    else:
        State[cell_ID] = 'cb_G1'
        cycleStartTime[cellID] = 0
        endofThisPhase[cellID] = getDuration(State[cell_ID])
        numDivisionsToDo[cellID] = divisions


def progressCycle():
    pass


def divideAndMutate():
    pass


# Algorithm 5 (Antigen Collection from FDCs)
def prgressFDCSelection():
    pass


# Algorithm 6 (Screening for T cell help at the Centrocyte Stage)
def progressTCellSelection():
    pass


# Algorithm 7 (Updating the T cells according to B cells Interactions)
def updateTCell():
    pass


def liberateTCell():
    pass


# Algorithm 8 (Transition between Centroblasts, centrocyctes, and output Cells)
def differToOut():
    pass


def differToCB():
    pass


def differ2CC():
    pass


# Algorithm 9 (Initialisation)
def initialiseCells():
    global cell_ID

    # Initialise Stromal Cells:
    for _ in range(NumStromalCells):
        pos = random.choice(DarkZone)  # Find empty location in dark zone
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
        # Find empty location in list zone
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
        x = pos[0];
        y = pos[1];
        z = pos[2]
        for i in range(1, DendriteLength + 1):
            # TODO implement O(1) checks for whether position is valid.
            for frag_pos in [(x + i, y, z), (x - i, y, z), (x, y + i, z), (x, y - i, z)]:
                if frag_pos in LightZone and Grid_ID[frag_pos] == None:
                    newID = cell_ID
                    cell_ID += 1
                    fragments.append(newID)
                    Position[newID] = pos
                    Grid_ID[frag_pos] = newID
                    Grid_Type[frag_pos] = 'Fragment'

                    # When Z axis is changing, we require extra check that we're still in light zone.
            for frag_pos in [(x, y, z + i), (x, y, z - i)]:
                if frag_pos in LightZone and Grid_ID[frag_pos] == None:
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
        CBDList.append(newID)
        Type[newID] = 'CENTROBLAST'
        BCR[newID] = None
        '''Need BCR values'''
        Position[newID] = pos
        pMutation[newID] = pMut(t)
        Grid_ID[pos] = newID
        Grid_Type[pos] = 'CENTROBLAST'

        initiate_cycle(newID, numDivFounderCells)
        initiate_chemokine_receptors(newID, 'CENTROBLAST')

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
        t += dt
        for StromalCell in StormaList:
            pass


# Extra Algorithms/Functions
def generateSpatialPoints(n):
    '''Obtains a list of all points in the sphere/GC
    Input: Number of discrete diamter points, n.
    Output: List containing all points within sphere.'''

    return [(x + n / 2 + 1, y + n / 2 + 1, z + n / 2 + 1) for x in range(-n / 2, n / 2) for y in range(-n / 2, n / 2)
            for z in range(-n / 2, n / 2) if ((x + 0.5) ** 2 + (y + 0.5) ** 2 + (z + 0.5) ** 2) <= (n / 2) ** 2]


def findEmptyNeighbour():
    pass


def pMut(time):
    '''Finds the probability of mutation
    Input: Current time, t
    Output: Probability of Mutation'''

    if time > 24:
        return 0.5
    else:
        return 0


def getDuration(cell_state):
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
        print("getDuration: Invalid cell state, {}".format(cell_state))

    return random.gauss(mu, sigma)


# Set-up for simulation:

# Distance Variables:
N = 64  # Diameter of sphere/GC
AllPoints = generateSpatialPoints(N)
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
CBDList = []
TCList = []

'''
Here we will create empty dictionaries to store different properties. Will add them as necessary.
'''
BCR = {}
pMutation = {}
Type = {}
State = {}
Position = {}
cycleStartTime = {}
endofThisPhase = {}
numDivisionsToDo = {}
responsiveToSignalCXCL12 = {}
responsiveToSignalCXCL13 = {}
Fragments = {}
FragmentAg = {}
FCellVol = {}

# Dictionaries storing what is at each location. Initially empty, so 'None'.
Grid_ID = {pos: None for pos in AllPoints}
Grid_Type = {pos: None for pos in AllPoints}

# Sequence variable for giving each cell an ID:
cell_ID = 0

# Dynamic number of divisions:
numDivFounderCells = 12

if __name__ == "__main__":

    initialiseCells()
    count = 0
    for i in AllPoints:
        if Grid_Type[i] is not None:
            count += 1
            print("Position = {}, Cell_id = {}, Cell_type = {}".format(i, Grid_ID[i], Grid_Type[i]))
    print(count)
    print(len(LightZone))




    #
