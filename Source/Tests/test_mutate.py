from Source.germinal_center import *

# Note that initialised cells have a probability of zero to mutate.
# Can use these initialised cells to test that the function does not
# mutate the cell when applied and event of mutation obviously does
# not occur.

def test_no_mutation():
    # Tests that nothing changes if no mutation occurs
    for _ in range(1):
        # Generate parameters and output
        parameters = Params()
        output = Out(parameters)

        initialise_cells(parameters, output)

        # Attempt to mutate cells - BCR shouldn't change
        for key in list(output.bcr.keys()):
            initial_bcr = output.bcr[key]
            mutate(key, parameters, output)
            assert initial_bcr == output.bcr[key], "Cell mutated even thought p_mutate = 0"


def test_mutation_hamming_dist():
    # We will test that hamming distance before and after mutation is one.
    for _ in range(1):
        # Generate parameters and output
        parameters = Params()
        output = Out(parameters)

        initialise_cells(parameters, output)

        # Apply mutation to cells
        for key in list(output.bcr.keys()):
            # Force mutation
            output.p_mutation[key] = 1.0
            initial_bcr = output.bcr[key]
            mutate(key, parameters, output)
            assert hamming_distance(initial_bcr, output.bcr[key]), f"Mutated bcr from {initial_bcr} to {output.bcr[key]}"


def test_mutation_mutating_9():
    # Testing cases where the value in the bcr being modified is a nine
    # Should become 8 as it can't be increased.

    for _ in range(1):
        # Generate parameters and output
        parameters = Params()
        output = Out(parameters)

        initialise_cells(parameters, output)

        # Apply mutation to cells
        for key in list(output.bcr.keys()):
            # Force mutation
            output.p_mutation[key] = 1.0
            output.bcr[key] = 9999  # Now value being modified must be 9
            mutate(key, parameters, output)
            assert output.bcr[key] in [8999, 9899, 9989, 9998], f"Mutated bcr from 9999 to {output.bcr[key]}"



def test_mutation_mutating_0():
    # Testing cases where the value in the bcr being modified is a nine
    # Should become 1 as it can't be decreased.
    # Also ensuring that if leading number is 1, then it will increase to two.

    for _ in range(1):
        # Generate parameters and output
        parameters = Params()
        output = Out(parameters)

        initialise_cells(parameters, output)

        # Apply mutation to cells
        for key in list(output.bcr.keys()):
            # Force mutation
            output.p_mutation[key] = 1.0
            output.bcr[key] = 1000  # Now value being modified must be 0 or leading 1.
            mutate(key, parameters, output)
            assert output.bcr[key] in [2000, 1100, 1010, 1001], f"Mutated bcr from 1000 to {output.bcr[key]}"