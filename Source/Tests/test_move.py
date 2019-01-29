from Source.germinal_center import *
import numpy as np

# TODO test with output states further along in development.

def test_not_move():
    for _ in range(1):
        parameters = Params()
        output = Out(parameters)
        initialise_cells(parameters, output)

        # Set speed for all moveable cells to zero
        # Forces probability of movement to be zero
        parameters.speed_centroblast = 0
        parameters.speed_centrocyte = 0
        parameters.speed_outcell = 0
        parameters.speed_tcell = 0

        moveable_cells = output.list_cb + output.list_cc + output.list_outcells + output.list_tc

        for cell in moveable_cells:
            initial_position = output.position[cell]
            move(cell, parameters, output)
            new_position = output.position[cell]
            assert initial_position == new_position, f"Cell moved from {initial_position} to {new_position}."

# We need helped functions before doing tests to check if movement occurs.
def check_neighbour(pos1, pos2):
    difference = np.abs(pos1 - pos2)
    # Check if all difference are zero or one
    # If not one, it's moved more than one square in a directiom
    return np.all((difference == 0) | (difference == 1))

# test helper function
def test_check_nieighbour1():
    pos1 = np.array([0,0,0])
    pos2 = np.array([0,0,0])
    assert check_neighbour(pos1, pos2),f"Failed test {pos1} and {pos2}"

def test_check_nieighbour2():
    pos1 = np.array([1,0,1])
    pos2 = np.array([0,0,0])
    assert check_neighbour(pos1, pos2),f"Failed test {pos1} and {pos2}"

def test_check_nieighbour3():
    pos1 = np.array([11, 5, 3])
    pos2 = np.array([12, 6, 2])
    assert check_neighbour(pos1, pos2),f"Failed test {pos1} and {pos2}"

def test_check_nieighbour4():
    pos1 = np.array([1,2,3])
    pos2 = np.array([4,5,6])
    assert not check_neighbour(pos1, pos2),f"Passed test {pos1} and {pos2}"

def test_check_nieighbour5():
    pos1 = np.array([0,0,2])
    pos2 = np.array([0,0,0])
    assert not check_neighbour(pos1, pos2),f"Passed test {pos1} and {pos2}"

# If movement fails, can be due to too many surrounding cells (>= 9)
# Following function tests if a function has >= 9 neighbours
def count_neighbours(cell, parameters, output):
    # Probably should test this function.
    pos = output.position[cell]
    counter = 0
    for transition in parameters.possible_neighbours:
        new_pos = tuple(np.array(pos) + np.array(transition))
        if output.grid_id[new_pos] is not None:
            counter += 1

    return True if counter > 8 else False


def test_move_appropriately():
    for _ in range(1):
        parameters = Params()
        output = Out(parameters)
        initialise_cells(parameters, output)

        # Set speed for all moveable cells to a large number
        # Forces probability of movement to be greater than one
        parameters.speed_centroblast = 1e20
        parameters.speed_centrocyte = 1e20
        parameters.speed_outcell = 1e20
        parameters.speed_tcell = 1e20

        moveable_cells = output.list_cb + output.list_cc + output.list_outcells + output.list_tc


        for cell in moveable_cells:
            initial_position = output.position[cell]
            move(cell, parameters, output)
            new_position = output.position[cell]
            if initial_position == new_position:
                assert count_neighbours(cell, parameters, output), f"Cell has less than 9 neighbours and didn't move"
            else:
                assert check_neighbour(np.array(initial_position), np.array(new_position)), f"Cell moved from {initial_position} to {new_position}"
