'''
Skeleton for Simulating a Germinal Center
'''

#Imports:
import random

#Algorithm 1 (Mutation)
def mutate():
    pass


#Algorithm 2 (Dynamic Updating of Chemotaxis)
def initiateChemokineReceptors(cell_ID, cell_type):
    if cell_type == 'Centroblast':
        responsiveToSignalCXCL12[cell_ID] = True
        responsiveToSignalCXCL13[cell_ID] = False
    elif cell_type == 'Centrocyte':
        responsiveToSignalCXCL12[cell_ID] = False
        responsiveToSignalCXCL13[cell_ID] = True
    elif cell_type == 'Outcell':
        responsiveToSignalCXCL12[cell_ID] = False
        responsiveToSignalCXCL13[cell_ID] = True
    else:
        print("initiateChemokineReceptors: Invalid cell_type, {}".format(cell_type))

def updateChemokinesReceptors():
    pass

#Algorithm 3 (Updating Position and Polarity of cells at each time-point)
def move():
    pass

def turnAngle():
    pass

#Algorithm 4 (Updating events at the Centroblast Stage)
def initiateCycle(cell_ID, divisions):
    if divisions == 0:
        State[cell_ID] = 'cb_stop_dividing'
    else:
        State[cell_ID] = 'cb_G1'
        cycleStartTime[cell_ID] = 0
        endofThisPhase[cell_ID] = getDuration(State[cell_ID])
        numDivisionsToDo[cell_ID] = divisions


def progressCycle():
    pass

def divideAndMutate():
    pass

#Algorithm 5 (Antigen Collection from FDCs)
def prgressFDCSelection():
    pass

#Algorithm 6 (Screening for T cell help at the Centrocyte Stage)
def progressTCellSelection():
    pass

#Algorithm 7 (Updating the T cells according to B cells Interactions)
def updateTCell():
    pass

def liberateTCell():
    pass

#Algorithm 8 (Transition between Centroblasts, centrocyctes, and output Cells)
def differToOut():
    pass

def differToCB():
    pass

def differ2CC():
    pass

#Algorithm 9 (Initialisation)
def initialiseCells():
    global cell_ID
    #Initialise Stromal Cells:
    for _ in range(NumStromalCells):
        pos = random.choice(DarkZone)       #Find empty location in dark zone
        while Grid_ID[pos] is not None:
            pos = random.choice(DarkZone)

        newID = cell_ID
        cell_ID += 1

        StormaList.append(newID)
        Type[newID] = 'Stromal'
        Grid_ID[pos] = newID
        Grid_type[pos] = 'Stromal'

    #Initialise Fragments:
    for _ in range(NumFDC):
        pos = random.choice(LightZone)      #Find empty location in list zone
        while Grid_ID[pos] is not None:
            pos = random.choice(LightZone)




    #Initialise Seeder Cells:
    for _ in range(NumSeeder):
        pos = random.choice(LightZone)      #Find empty location in list zone
        while Grid_ID[pos] is not None:
            pos = random.choice(LightZone)

        newID = cell_ID
        cell_ID += 1

        CBDList.append(newID)
        Type[newID] = 'Centroblast'
        BCR[newID] = None
        '''Need BCR values'''
        pMutation[newID] = pMut(t)
        Grid_ID[pos] = newID
        Grid_type[pos] = 'Centroblast'

        initiateCycle(newID, numDivFounderCells)
        initiateChemokineReceptors(newID, 'Centroblast')

    #Initialise T Cells:
    for _ in range(NumTC):
        pos = random.choice(LightZone)       #Find empty location in light zone
        while Grid_ID[pos] is not None:
            pos = random.choice(LightZone)

        newID = cell_ID
        cell_ID += 1

        TCList.append(newID)
        Type[newID] = 'TCell'
        Grid_ID[pos] = newID
        Grid_type[pos] = 'TCell'



#Algorithm 10 (Hyphasma: Simulation of Germinal Center)
def hyphasma():
    pass

#Extra Algorithms/Functions
def generateSpatialPoints(n):
    '''Obtains a list of all points in the sphere/GC
    Input: Number of discrete diamter points, n.
    Output: List containing all points within sphere.'''

    return [(x+n/2+1, y+n/2+1, z+n/2+1) for x in range(-n/2, n/2) for y in range(-n/2, n/2) for z in range(-n/2, n/2) if ((x + 0.5)**2 + (y + 0.5)**2 + (z + 0.5)**2) <= (n/2)**2]

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
    sd = 1
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

    return random.gauss(mu, sd)

if __name__ == "__main__":
    #Set-up for simulation:

    #Distance Variables:
    N = 16      #Diameter of sphere/GC
    AllPoints = generateSpatialPoints(N)
    DarkZone = [point for point in AllPoints if point[2] > N/2 - 1]
    LightZone = [point for point in AllPoints if point[2] <= N/2 - 1]

    dx = 5

    #Time Variables:
    dt = 0.002
    tmin = 0.0
    tmax = 504.0
    t = 0.0     #Current time

    #Initialisation
    NumStromalCells = 300
    NumFDC = 200
    NumSeeder = 3
    NumTC = 250

    DendriteLength = 8

    #Lists to store ID of each cell in each state (and fragments)
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
    cycleStartTime = {}
    endofThisPhase = {}
    numDivisionsToDo = {}
    responsiveToSignalCXCL12 = {}
    responsiveToSignalCXCL13 = {}


    #Dictionaries storing what is at each location. Initially empty, so 'None'.
    Grid_ID = {pos:None for pos in AllPoints}
    Grid_type = {pos:None for pos in AllPoints}

    #Sequence variable for giving each cell an ID:
    cell_ID = 0


    #Dynamic number of divisions:
    numDivFounderCells = 12




    #
