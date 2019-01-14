from Source.germinal_center import *

"""
Very important function to test - many other tests rely on this function.
Will test general properties such as:
    - Correct number of each cell generated
    - Appropriate values for each property
    - Only one cell at each discrete location
    - Each cell has a unique cell id
    - Type of cell at position is recorded
"""

def test_stromal():
    for _ in range(1):
        # Generate parameters and output
        parameters = Params()
        output = Out(parameters)

        initialise_cells(parameters, output)
        for cell in output.list_stromal:
            pos = output.position[cell]
            assert output.type[cell] == CellType.Stromal, f"Cell type is {output.type[cell]}, not Stromal"
            assert output.grid_type[pos] == CellType.Stromal, f"Cell grid_type is {output.grid_type[pos]}, not Stromal"
            assert output.grid_id[pos] == cell, f"Wrong cell id at location {pos}: {output.grid_id[pos]} instead of {cell}"
            assert pos in parameters.dark_zone, f"Cell is at {pos}, outside dark zone"

        assert len(output.list_stromal) == parameters.initial_num_stromal_cells, f"{len(output.list_stromal)} Stromal cells, not {parameters.initial_num_stromal_cells}"


def test_centrocytes():
    for _ in range(1):
        # Generate parameters and output
        parameters = Params()
        output = Out(parameters)

        initialise_cells(parameters, output)

        assert len(output.list_cc) == 0, f"Non-zero number of centrocytes: {output.list_cc}"


def test_T_cells():
    for _ in range(10):
        # Generate parameters and output
        parameters = Params()
        output = Out(parameters)

        initialise_cells(parameters, output)

        for cell in output.list_tc:
            pos = output.position[cell]
            assert output.type[cell] == CellType.TCell, f"Cell type is {output.type[cell]}, not T cell"
            assert output.grid_type[pos] == CellType.TCell, f"Cell grid_type is {output.grid_type[pos]}, not T cell"
            assert output.grid_id[pos] == cell, f"Wrong cell id at location {pos}: {output.grid_id[pos]} instead of {cell}"
            assert pos in parameters.light_zone, f"Cell is at {pos}, outside light zone"

        assert len(output.list_tc) == parameters.initial_num_tcells, f"Incorrect number of t cells: {len(output.list_tc)}, not {parameters.initial_num_tcells} "

def test_centroblasts():
    for _ in range(1):
        # Generate parameters an d output
        parameters = Params()
        output = Out(parameters)

        initialise_cells(parameters, output)

        for cell in output.list_cb:
            pos = output.position[cell]
            assert output.type[cell] == CellType.Centroblast, f"Cell type is {output.type[cell]}, not cb cell"
            assert output.grid_type[pos] == CellType.Centroblast, f"Cell grid_type is {output.grid_type[pos]}, not cb cell"
            assert output.grid_id[pos] == cell, f"Wrong cell id at location {pos}: {output.grid_id[pos]} instead of {cell}"
            assert pos in parameters.light_zone, f"Cell is at {pos}, outside light zone"

            assert output.bcr[cell] in parameters.bcr_values_initial, f"cb assigned bcr: {output.bcr[cell]} which is in available bcr list: {parameters.bcr_values_initial}"

            # Partial test of initiate_chemokine_receptors:
            assert output.responsive_to_cxcl12[cell] == True, "Responsive to cxcl12 is False"
            assert output.responsive_to_cxcl13[cell] == False, "Responsive to cxcl13 is True"

        assert len(output.list_cb) == parameters.initial_num_seeder, f"Incorrect number of cb cells: {len(output.list_cb)}, not {parameters.initial_num_seeder} "



def test_f_cells():
    for _ in range(1):
        # Generate parameters an d output
        parameters = Params()
        output = Out(parameters)

        initialise_cells(parameters, output)
        for cell in output.list_fdc:
            # Check properties of F cell
            pos = output.position[cell]
            assert output.type[cell] == CellType.FCell, f"Cell type is {output.type[cell]}, not F cell"
            assert output.grid_type[pos] == CellType.FCell, f"Cell grid_type is {output.grid_type[pos]}, not F cell"
            assert output.grid_id[pos] == cell, f"Wrong cell id at location {pos}: {output.grid_id[pos]} instead of {cell}"
            assert pos in parameters.light_zone, f"Cell is at {pos}, outside light zone"
            assert output.ic_amount[cell] >= 0, f"negative ic_amount {output.ic_amount[cell]}"

            # Check properties of associated Fragments
            assert len(output.fragments[cell]) <= 6 * parameters.dendrite_length, f"{len(output.fragments[cell])} fragments, more than maximum {6*parameters.dendrite_length}"

            for frag in output.fragments[cell]:
                assert output.type[frag] == CellType.Fragment, f"Cell type is {output.type[cell]}, not Fragment"
                assert output.grid_type[pos] == CellType.FCell, f"Cell grid_type is {output.grid_type[pos]}, not Fragment"
                assert output.grid_id[pos] == cell, f"Wrong cell id at location {pos}: {output.grid_id[pos]} instead of {cell}"
                assert output.parent[frag] == cell, f"Wrong parent, {output.parent[frag]} instead of {cell}"
                assert output.ic_amount[frag] >= 0, f"negative ic_amount {output.ic_amount[cell]}"


        assert len(output.list_fdc) == parameters.initial_num_fdc, f"Incorrect number of t cells: {len(output.list_fdc)}, not {parameters.initial_num_fdc} "



def test_general():
    for _ in range(10):
        # Generate parameters an d output
        parameters = Params()
        output = Out(parameters)

        initialise_cells(parameters, output)
        # Make sure no two cells have same id
        list_all_cells = output.list_fdc + output.list_tc + output.list_cb + output.list_cc + output.list_stromal + output.list_outcells
        list_all_cells += [item for sublist in output.fragments.values() for item in sublist]

        assert len(list_all_cells) == len(set(list_all_cells)), f"Duplicate ids assigned. No. of cells: {len(list_all_cells)}, No. of ids: {len(set(list_all_cells))} "

        # Make sure no two cells have same location
        assert len(output.position.values()) == len(set(output.position.values())), f"Duplicate ids assigned. No. of cells: {len(output.position.values())}, No. of ids: {len(set(output.position.values()))} "
