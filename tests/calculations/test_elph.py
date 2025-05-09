# -*- coding: utf-8 -*-
"""Tests for the `ElPhCalculation` class."""
import os

from aiida import orm
from aiida.common import datastructures
from aiida.plugins import CalculationFactory
import pytest

ElPhCalculation = CalculationFactory('vibroscopy.elph')


@pytest.fixture
def generate_inputs_elph(
    fixture_code,
    generate_remote_folder,
    generate_phonopy_data,
):
    """Return a minimal input for the `ElPhCalculation`."""

    def _generate_inputs_elph(parameters=None, phonopy_options={}):
        """Return a minimal input for the `ElPhCalculation`."""
        phonopy_data = generate_phonopy_data(**phonopy_options)

        # Generating parameters
        if parameters is None:
            parameters = orm.Dict({'mpi': 4})

        # Building input dict to return
        ret_dic = {
            'code': fixture_code('vibroscopy.elph'),
            'parent_nscf_folders': {
                'scf_0': generate_remote_folder(),
                'group_1': generate_remote_folder(),
            },
            'phonopy_data': phonopy_data,
            'parameters': parameters,
            # 'metadata': {
            #     'options': get_default_options()
            # },
        }

        return ret_dic

    return _generate_inputs_elph


# def test_default(fixture_sandbox_folder, generate_calc_job, generate_inputs_elph, file_regression):
#     """Test a default `ElPhCalculation`."""
#     import os

#     remote_copy_list = []
#     for folder_final, parent in self.inputs.parent_nscf_folders.items():
#         folder_src = os.path.join(parent.get_remote_path(), '*')
#         remote_copy_list.append((parent.computer.uuid, folder_src, folder_final))
#     entry_point_name = 'vibroscopy.elph'

#     calc_info = generate_calc_job(fixture_sandbox_folder, entry_point_name, generate_inputs_elph())

#     filename_input = ElPhCalculation.spec().inputs.get_port('metadata.options.input_filename').default
#     filename_output = ElPhCalculation.spec().inputs.get_port('metadata.options.output_filename').default

#     cmdline_params = [filename_input]
#     retrieve_list = [filename_output]

#     # computer =

#     # Check the attributes of the returned `CalcInfo`
#     assert isinstance(calc_info, datastructures.CalcInfo)
#     assert sorted(calc_info.codes_info[0].cmdline_params) == sorted(cmdline_params)
#     assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)
#     assert sorted(calc_info.remote_copy_list) == sorted(remote_copy_list)
#     assert calc_info.local_copy_list is None

#     with fixture_sandbox_folder.open(filename_input) as handle:
#         input_written = handle.read()

#     # Checks on the files written to the sandbox folder as raw input
#     assert sorted(fixture_sandbox_folder.get_content_list()) == sorted([filename_input])
#     file_regression.check(input_written, encoding='utf-8', extension='.json')

# def test_settings(fixture_sandbox_folder, generate_calc_job, generate_inputs_elph):
#     """Test a default `ElPhCalculation` with `settings` in inputs."""
#     entry_point_name = 'vibroscopy.elph'

#     inputs = generate_inputs_elph()
#     cmdline_params = ['-nk', '4', '-nband', '2', '-ntg', '3', '-ndiag', '12']
#     inputs['settings'] = orm.Dict({'cmdline': cmdline_params})
#     calc_info = generate_calc_job(fixture_sandbox_folder, entry_point_name, inputs)

#     # Check that the command-line parameters are as expected.
#     assert calc_info.codes_info[0].cmdline_params == cmdline_params + ['aiida.in']
