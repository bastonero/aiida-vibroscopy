# -*- coding: utf-8 -*-
"""CalcJob for phonopy post-processing."""
from aiida import orm
from aiida.common import datastructures
from aiida.engine import CalcJob
from aiida_phonopy.data import PhonopyData
from aiida_phonopy.utils.mapping import _lowercase_dict
import numpy as np

__all__ = ('ElPhCalculation',)


def validate_phonopy_data(value, _):
    """Validate that PhonopyData is compatibly with ElectronPhonon.jl.

    Currently, the code is expecting diagonal supercells with the same dimensions.
    """
    supercell_matrix = np.array(value.supercell_matrix)

    if not supercell_matrix[0][0] == supercell_matrix[1][1] == supercell_matrix[2][2]:
        return 'The supercell matrix must have equal diagonal elements.'

    if not np.isclose(np.diag(supercell_matrix), supercell_matrix, atol=1.0e-5):
        return 'The supercell matrix must be diagonal.'


def validate_parent_nscf_folders(_):
    """Validate the `parent_nscf_folders` namespace.

    Verify that the keys respect the convention of ElectronPhonon.jl.
    """
    return None


class ElPhCalculation(CalcJob):
    """Base `CalcJob` implementation for ElectronPhonon.jl.

    The crucial step is to set correctly the parent NSCF folders. The `parent_nscf_folders`
    dictionary must be set with the following convention:
    - folders for the displaced structures need to be specified as `group_I`, I is the
      corresponding displacement index as given by Phonopy
    - the folder for the SCF in unit cell must be specified as `scf_0`
    """

    _DEFAULT_INPUT_FILE = 'aiida.json'
    _DEFAULT_OUTPUT_FILE = 'aiida.out'
    _DEFAULT_PHONOPY_FILE = 'phonopy_params.yaml'

    _INPUT_SUBFOLDER = './'
    _OUTPUT_SUBFOLDER = './'

    @classmethod
    def define(cls, spec):
        """Define inputs, outputs, and outline."""
        super().define(spec)

        # yapf: disable
        spec.input('parameters', valid_type=orm.Dict, required=True)
        spec.input('phonopy_data', valid_type=PhonopyData, required=True, validator=validate_phonopy_data,
            help=('Data node containing the displacements as well as forces '
            'and non-analytical constants info of a previous.'))
        spec.input_namespace('parent_nscf_folders', valid_type=orm.RemoteData,
            required=True, validator=validate_parent_nscf_folders,
            help='Mapping, i.e. a dictionary, of parent NSCF folders produced by PwCalculation.')
        spec.input('settings', valid_type=orm.Dict, required=False,
            help=('Settings for the ElectronPhonon.jl calculation. It can contain one of the '
                  'following keys: `cmdline` (list), `symmetrize_nac` (bool), '
                  '`factor_nac` (float), `subtract_residual_forces` (bool)'),)

        spec.input('metadata.options.withmpi', valid_type=bool, default=False)
        spec.input('metadata.options.input_filename', valid_type=str, default=cls._DEFAULT_INPUT_FILE)
        spec.input('metadata.options.output_filename', valid_type=str, default=cls._DEFAULT_OUTPUT_FILE)
        spec.input('metadata.options.parser_name', valid_type=str, default='vibroscopy.elph')
        spec.inputs['metadata']['options']['resources'].default = lambda: {'num_machines': 1}

        spec.output('parameters', valid_type=orm.Dict, required=False,
            help='Summary of the ElectronPhonon.jl calculation.')

        # Unrecoverable errors: required retrieved files could not be read, parsed or are otherwise incomplete
        spec.exit_code(301, 'ERROR_NO_RETRIEVED_TEMPORARY_FOLDER',
            message='The retrieved temporary folder could not be accessed.')
        spec.exit_code(302, 'ERROR_OUTPUT_STDOUT_MISSING',
            message='The retrieved folder did not contain the required stdout output file.')
        spec.exit_code(304, 'ERROR_OUTPUT_FILES_MISSING',
            message='The retrieved folder did not contain one or more expected output files.')

        spec.exit_code(310, 'ERROR_OUTPUT_STDOUT_READ',
            message='The stdout output file could not be read.')
        spec.exit_code(311, 'ERROR_OUTPUT_STDOUT_PARSE',
            message='The stdout output file could not be parsed.')
        spec.exit_code(312, 'ERROR_OUTPUT_STDOUT_INCOMPLETE',
            message='The stdout output file was incomplete probably because the calculation got interrupted.')

        spec.exit_code(350, 'ERROR_UNEXPECTED_PARSER_EXCEPTION',
            message='The parser raised an unexpected exception.')
        # yapf: enable

    def prepare_for_submission(self, folder):
        """Prepare the calculation job for submission by transforming input nodes into input files.

        In addition to the input files being written to the sandbox folder, a `CalcInfo` instance will be returned that
        contains lists of files that need to be copied to the remote machine before job submission, as well as file
        lists that are to be retrieved after job completion.

        :param folder: a sandbox folder to temporarily write files on disk.
        :return: :py:class:`~aiida.common.datastructures.CalcInfo` instance.
        """
        settings = {}
        if 'settings' in self.inputs:
            settings = _lowercase_dict(self.inputs.settings.get_dict(), dict_name='settings')

        # ================= prepare the elph input files ===================
        self.write_phonopy_info(folder)

        parameters = _lowercase_dict(self.inputs.parameters.get_dict(), dict_name='parameters')
        filename = self.options.input_filename
        self.write_calculation_input(folder, parameters, filename)

        # ================= retreiving elph output files ===================
        retrieve_list = [self.options.output_filename]
        retrieve_temporary_list = []

        # ============================ calcinfo ===============================
        local_copy_list = []

        calcinfo = datastructures.CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.local_copy_list = local_copy_list  # what we write in the folder
        calcinfo.remote_copy_list = self.get_remote_copy_list()
        calcinfo.retrieve_list = retrieve_list  # what to retrieve and keep
        calcinfo.retrieve_temporary_list = retrieve_temporary_list

        codeinfo = datastructures.CodeInfo()  # new code info object per cmdline
        codeinfo.cmdline_params = (list(settings.pop('cmdline', [])) + [filename])
        codeinfo.stdout_name = self.options.output_filename
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.withmpi = self.options.withmpi
        calcinfo.codes_info.append(codeinfo)

        return calcinfo

    def write_phonopy_info(self, folder):
        """Write in `folder` the `phonopy.yaml` file."""
        from phonopy.interface.phonopy_yaml import PhonopyYaml

        kwargs = {}

        if 'settings' in self.inputs:
            the_settings = self.inputs.settings.get_dict()
            for key in ['symmetrize_nac', 'factor_nac', 'subtract_residual_forces']:
                if key in the_settings:
                    kwargs.update({key: the_settings[key]})

        ph = self.inputs.phonopy_data.get_phonopy_instance(**kwargs)

        # Setting the phonopy yaml object to produce yaml lines
        # .. note: this does not write the force constants
        phpy_yaml = PhonopyYaml()
        phpy_yaml.set_phonon_info(ph)
        phpy_yaml_txt = str(phpy_yaml)

        with folder.open(self._DEFAULT_PHONOPY_FILE, 'w', encoding='utf8') as handle:
            handle.write(phpy_yaml_txt)

    def write_calculation_input(self, folder, parameters: dict, filename: str):
        """Write in `folder` the input file containing the information regarding the calculation."""
        import json

        with folder.open(filename, 'w', encoding='utf8') as handle:
            json.dump(handle, parameters)

    def get_remote_copy_list(self) -> list[tuple]:
        """Return the `remote_copy_list`.

        :returns: list of resource copy instructions
        """
        import os

        remote_copy_list = []
        for folder_final, parent in self.inputs.parent_nscf_folders.items():
            folder_src = os.path.join(parent.get_remote_path(), '*')
            remote_copy_list.append((parent.computer.uuid, folder_src, folder_final))

        return remote_copy_list
