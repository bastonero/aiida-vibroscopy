# -*- coding: utf-8 -*-
#################################################################################
# Copyright (c), All rights reserved.                                           #
# This file is part of the AiiDA-Vibroscopy code.                               #
#                                                                               #
# The code is hosted on GitHub at https://github.com/bastonero/aiida-vibroscopy #
# For further information on the license, see the LICENSE.txt file              #
#################################################################################
# pylint: disable=redefined-outer-name
"""Initialise a text database and profile for pytest."""
import io
import os
import pathlib
import tempfile

import pytest

pytest_plugins = ['aiida.manage.tests.pytest_fixtures']  # pylint: disable=invalid-name


@pytest.fixture(scope='session')
def filepath_tests():
    """Return the absolute filepath of the `tests` folder.

    .. warning:: if this file moves with respect to the `tests` folder, the implementation should change.

    :return: absolute filepath of `tests` folder which is the basepath for all test resources.
    """
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def filepath_fixtures(filepath_tests):
    """Return the absolute filepath to the directory containing the file `fixtures`."""
    return os.path.join(filepath_tests, 'fixtures')


@pytest.fixture
def filepath_fixtures(filepath_tests):
    """Return the absolute filepath to the directory containing the file `fixtures`."""
    return os.path.join(filepath_tests, 'fixtures')


@pytest.fixture(scope='function')
def fixture_sandbox_folder():
    """Return a `SandboxFolder`."""
    from aiida.common.folders import SandboxFolder
    with SandboxFolder() as folder:
        yield folder


@pytest.fixture
def fixture_localhost(aiida_localhost):
    """Return a localhost `Computer`."""
    localhost = aiida_localhost
    localhost.set_default_mpiprocs_per_machine(1)
    return localhost


@pytest.fixture
def fixture_code(fixture_localhost):
    """Return a `Code` instance configured to run calculations of given entry point on localhost `Computer`."""

    def _fixture_code(entry_point_name):
        from aiida.common import exceptions
        from aiida.orm import InstalledCode, load_code

        label = f'test.{entry_point_name}'

        try:
            return load_code(label=label)
        except exceptions.NotExistent:
            return InstalledCode(
                label=label,
                computer=fixture_localhost,
                filepath_executable='/bin/true',
                default_calc_job_plugin=entry_point_name,
            )

    return _fixture_code


@pytest.fixture
def serialize_builder():
    """Serialize the given process builder into a dictionary with nodes turned into their value representation.

    :param builder: the process builder to serialize
    :return: dictionary
    """

    def serialize_data(data):
        # pylint: disable=too-many-return-statements
        from aiida.orm import AbstractCode, BaseType, Data, Dict, KpointsData, List, RemoteData, SinglefileData
        from aiida.plugins import DataFactory

        StructureData = DataFactory('core.structure')
        UpfData = DataFactory('pseudo.upf')

        if isinstance(data, dict):
            return {key: serialize_data(value) for key, value in data.items()}

        if isinstance(data, BaseType):
            return data.value

        if isinstance(data, AbstractCode):
            return data.full_label

        if isinstance(data, Dict):
            return data.get_dict()

        if isinstance(data, List):
            return data.get_list()

        if isinstance(data, StructureData):
            return data.get_formula()

        if isinstance(data, UpfData):
            return f'{data.element}<md5={data.md5}>'

        if isinstance(data, RemoteData):
            # For `RemoteData` we compute the hash of the repository. The value returned by `Node._get_hash` is not
            # useful since it includes the hash of the absolute filepath and the computer UUID which vary between tests
            return data.base.repository.hash()

        if isinstance(data, KpointsData):
            try:
                return data.get_kpoints()
            except AttributeError:
                return data.get_kpoints_mesh()

        if isinstance(data, SinglefileData):
            return data.get_content()

        if isinstance(data, Data):
            return data.base.caching._get_hash()  # pylint: disable=protected-access

        return data

    def _serialize_builder(builder):
        return serialize_data(builder._inputs(prune=True))  # pylint: disable=protected-access

    return _serialize_builder


@pytest.fixture(scope='session', autouse=True)
def sssp(aiida_profile, generate_upf_data):
    """Create an SSSP pseudo potential family from scratch."""
    from aiida.common.constants import elements
    from aiida.plugins import GroupFactory

    aiida_profile.clear_profile()

    SsspFamily = GroupFactory('pseudo.family.sssp')

    cutoffs = {}
    stringency = 'standard'

    with tempfile.TemporaryDirectory() as dirpath:
        for values in elements.values():

            element = values['symbol']

            actinides = ('Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr')

            if element in actinides:
                continue

            upf = generate_upf_data(element)
            dirpath = pathlib.Path(dirpath)
            filename = dirpath / f'{element}.upf'

            with open(filename, 'w+b') as handle:
                with upf.open(mode='rb') as source:
                    handle.write(source.read())
                    handle.flush()

            cutoffs[element] = {
                'cutoff_wfc': 30.0,
                'cutoff_rho': 240.0,
            }

        label = 'SSSP/1.2/PBEsol/efficiency'
        family = SsspFamily.create_from_folder(dirpath, label)

    family.set_cutoffs(cutoffs, stringency, unit='Ry')

    return family


@pytest.fixture
def generate_workchain():
    """Generate an instance of a `WorkChain`."""

    def _generate_workchain(entry_point, inputs):
        """Generate an instance of a `WorkChain` with the given entry point and inputs.

        :param entry_point: entry point name of the work chain subclass.
        :param inputs: inputs to be passed to process construction.
        :return: a `WorkChain` instance.
        """
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager
        from aiida.plugins import WorkflowFactory

        process_class = WorkflowFactory(entry_point)
        runner = get_manager().get_runner()
        process = instantiate_process(runner, process_class, **inputs)

        return process

    return _generate_workchain


@pytest.fixture
def generate_structure():
    """Return a `StructureData` representing bulk silicon."""

    def _generate_structure(structure_id=None, hubbard=False):
        """Return a `StructureData` representing bulk silicon."""
        from aiida.orm import StructureData
        from aiida_quantumespresso.data.hubbard_structure import HubbardStructureData

        if structure_id is None:
            name = 'O'
            cell = [[3.9625313477, -3.9625313477, 0.0], [-3.9625313477, 0.0, 3.9625313477],
                    [0.0, -3.9625313477, -3.9625313477]]
            structure = StructureData(cell=cell)
            structure.append_atom(position=(0., 0., 0.), symbols='Mg', name='Mg')
            structure.append_atom(position=(1.98126567385, 1.98126567385, 1.98126567385), symbols='O', name='O')

        if structure_id == 'silicon':
            name = 'Si'
            param = 5.43
            cell = [[param / 2., param / 2., 0], [param / 2., 0, param / 2.], [0, param / 2., param / 2.]]
            structure = StructureData(cell=cell)
            structure.append_atom(position=(0., 0., 0.), symbols='Si', name='Si')
            structure.append_atom(position=(param / 4., param / 4., param / 4.), symbols='Si', name='Si')

        if hubbard:
            structure = HubbardStructureData.from_structure(structure)
            structure.initialize_onsites_hubbard(name, '2p')

        return structure

    return _generate_structure


@pytest.fixture
def generate_preprocess_data(generate_structure):
    """Generate a `PreProcessData`."""

    def _generate_preprocess_data(supercell_matrix=None, primitive_matrix=None):
        """Return a `PreProcessData` with bulk silicon as structure."""
        from aiida.plugins import DataFactory

        PreProcessData = DataFactory('phonopy.preprocess')
        structure = generate_structure()

        if supercell_matrix is None:
            supercell_matrix = [1, 1, 1]
        if primitive_matrix is None:
            primitive_matrix = 'auto'

        preprocess_data = PreProcessData(
            structure=structure, supercell_matrix=supercell_matrix, primitive_matrix=primitive_matrix
        )

        return preprocess_data

    return _generate_preprocess_data


@pytest.fixture
def generate_kpoints_mesh():
    """Return a `KpointsData` node."""

    def _generate_kpoints_mesh(npoints):
        """Return a `KpointsData` with a mesh of npoints in each direction."""
        from aiida.orm import KpointsData

        kpoints = KpointsData()
        kpoints.set_kpoints_mesh([npoints] * 3)

        return kpoints

    return _generate_kpoints_mesh


@pytest.fixture
def generate_trajectory():
    """Return a `TrajectoryData` node."""

    def _generate_trajectory(trajectory=None):
        """Return a `TrajectoryData` with minimum data."""
        from aiida.orm import TrajectoryData
        import numpy as np

        if trajectory is None:
            node = TrajectoryData()
            node.set_array('electronic_dipole_cartesian_axes', np.array([[[0., 0., 0.]]]))
            node.set_array('forces', np.array([[[0., 0., 0.], [0., 0., 0.]]]))
            stepids = np.array([1])
            times = stepids * 0.0
            cells = np.array([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
            positions = np.array([[[0., 0., 0.], [0., 0., 0.]]])
            node.set_trajectory(stepids=stepids, cells=cells, symbols=['Mg', 'O'], positions=positions, times=times)

        return node.store()

    return _generate_trajectory


@pytest.fixture
def generate_inputs_pw(fixture_code, generate_structure, generate_kpoints_mesh, generate_upf_data):
    """Generate default inputs for a `PwCalculation."""

    def _generate_inputs_pw():
        """Generate default inputs for a `PwCalculation."""
        from aiida.orm import Dict
        from aiida_quantumespresso.utils.resources import get_default_options

        parameters = Dict({
            'CONTROL': {
                'calculation': 'scf'
            },
            'SYSTEM': {
                'ecutrho': 400.0,
                'ecutwfc': 50.0
            },
            'ELECTRONS': {
                'mixing_beta': 0.4
            },
        })
        structure = generate_structure()
        inputs = {
            'code': fixture_code('quantumespresso.pw'),
            'structure': structure,
            'kpoints': generate_kpoints_mesh(2),
            'parameters': parameters,
            'pseudos': {kind: generate_upf_data(kind) for kind in structure.get_kind_names()},
            'metadata': {
                'options': get_default_options()
            }
        }

        return inputs

    return _generate_inputs_pw


@pytest.fixture
def generate_inputs_phonopy(fixture_code):
    """Generate default inputs for a `PhonopyCalculation."""

    def _generate_inputs_pw():
        """Generate default inputs for a `PhonopyCalculation."""
        from aiida.orm import Dict
        from aiida_quantumespresso.utils.resources import get_default_options

        parameters = Dict({'bands': 'auto'})
        inputs = {
            'code': fixture_code('phonopy.phonopy'),
            'parameters': parameters,
            'metadata': {
                'options': get_default_options()
            }
        }

        return inputs

    return _generate_inputs_pw


@pytest.fixture
def generate_inputs_dielectric(generate_inputs_pw):
    """Generate default inputs for a `DielectricWorkChain`."""

    def _generate_inputs_dielectric(
        property='raman', clean_workdir=True, electric_field_step=None, accuracy=None, diagonal_scale=None, **_
    ):
        """Generate default inputs for a `DielectricWorkChain`."""
        from aiida.common.extendeddicts import AttributeDict
        from aiida.orm import Bool, Float, Int
        inputs_scf = generate_inputs_pw()

        kpoints = inputs_scf.pop('kpoints')

        inputs = {
            'property': property,
            'scf': {
                'pw': inputs_scf,
                'kpoints': kpoints,
            },
            'clean_workdir': Bool(clean_workdir),
            'settings': {
                'sleep_submission_time': 0.
            },
            'central_difference': {},
            'symmetry': {},
        }

        if electric_field_step is not None:
            inputs['central_difference']['electric_field_step'] = Float(electric_field_step)
        if accuracy is not None:
            inputs['central_difference']['accuracy'] = Int(accuracy)
        if diagonal_scale is not None:
            inputs['central_difference']['diagonal_scale'] = Float(diagonal_scale)

        return AttributeDict(inputs)

    return _generate_inputs_dielectric


@pytest.fixture(scope='session')
def generate_upf_data():
    """Return a `UpfData` instance for the given element a file for which should exist in `tests/fixtures/pseudos`."""

    def _generate_upf_data(element):
        """Return `UpfData` node."""
        from aiida_pseudo.data.pseudo import UpfData
        content = f'<UPF version="2.0.1"><PP_HEADER\nelement="{element}"\nz_valence="4.0"\n/></UPF>\n'
        stream = io.BytesIO(content.encode('utf-8'))
        return UpfData(stream, filename=f'{element}.upf')

    return _generate_upf_data


@pytest.fixture
def generate_inputs_pw_base(generate_inputs_pw):
    """Generate default inputs for a `PwBaseWorkChain`."""

    def _generate_inputs_pw_base():
        """Generate default inputs for a `PwBaseWorkChain`."""
        inputs_scf = generate_inputs_pw()

        kpoints = inputs_scf.pop('kpoints')

        inputs = {
            'pw': inputs_scf,
            'kpoints': kpoints,
        }

        return inputs

    return _generate_inputs_pw_base


@pytest.fixture
def generate_base_scf_workchain_node(fixture_localhost, generate_trajectory):
    """Generate an instance of `WorkflowNode`."""

    def _generate_base_scf_workchain_node(exit_status=0, with_trajectory=False):
        from aiida import orm
        from aiida.common import LinkType
        from aiida.plugins.entry_point import format_entry_point_string
        from plumpy import ProcessState

        node = orm.WorkflowNode().store()
        node.set_process_state(ProcessState.FINISHED)
        node.set_exit_status(exit_status)

        # Add output Dict
        parameters = orm.Dict({
            'occupations': 'fixed',
            'number_of_bands': 5,
            'total_magnetization': 1,
            'volume': 10,
        }).store()
        parameters.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label='output_parameters')

        # Add a CalcJob node with RemoteData
        entry_point = format_entry_point_string('aiida.calculations', 'quantumespresso.pw')
        calcjob_node = orm.CalcJobNode(computer=fixture_localhost, process_type=entry_point)
        calcjob_node.store()

        remote_folder = orm.RemoteData(computer=fixture_localhost, remote_path='/tmp').store()
        remote_folder.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label='remote_folder')
        remote_folder.base.links.add_incoming(calcjob_node, link_type=LinkType.CREATE, link_label='remote_folder')
        remote_folder.store()

        # Add TrajectoryData output
        if with_trajectory:
            trajectory = generate_trajectory()
            trajectory.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label='output_trajectory')

        return node

    return _generate_base_scf_workchain_node


@pytest.fixture
def generate_dielectric_workchain_node():
    """Generate an instance of `WorkflowNode`."""

    def _generate_dielectric_workchain_node(raman=False,):
        from aiida import orm
        from aiida.common import LinkType
        import numpy

        labels = ['numerical_accuracy_2_step_1', 'numerical_accuracy_4']

        node = orm.WorkflowNode().store()

        for label in labels:
            tensors = orm.ArrayData()

            dielectric_array = numpy.eye(3)
            tensors.set_array('dielectric', dielectric_array)

            b = numpy.eye(3)
            born_charges_array = numpy.array([b, -b])
            tensors.set_array('born_charges', born_charges_array)

            if raman:
                r = numpy.zeros((3, 3, 3))
                numpy.fill_diagonal(r, 1)
                dph0_array = numpy.array([r, -r])
                tensors.set_array('raman_tensors', dph0_array)

                nlo_array = numpy.zeros((3, 3, 3))
                tensors.set_array('nlo_susceptibility', nlo_array)

            tensors.store()

            tensors.base.links.add_incoming(node, link_type=LinkType.RETURN, link_label=f'tensors__{label}')

        return node

    return _generate_dielectric_workchain_node


@pytest.fixture
def generate_vibrational_data_from_forces(generate_structure):
    """Generate a `VibrationalFrozenPhononData`."""

    def _generate_vibrational_data_from_forces(dielectric=None, born_charges=None, dph0=None, nlo=None, forces=None):
        """Return a `VibrationalFrozenPhononData` with bulk silicon as structure."""
        from aiida.plugins import DataFactory
        import numpy

        VibrationalFrozenPhononData = DataFactory('vibroscopy.fp')
        PreProcessData = DataFactory('phonopy.preprocess')

        structure = generate_structure()

        supercell_matrix = [1, 1, 1]

        preprocess_data = PreProcessData(
            structure=structure, supercell_matrix=supercell_matrix, primitive_matrix='auto'
        )

        preprocess_data.set_displacements()

        vibrational_data = VibrationalFrozenPhononData(preprocess_data=preprocess_data)

        if dielectric is not None:
            vibrational_data.set_dielectric(dielectric)
        else:
            vibrational_data.set_dielectric(numpy.eye(3))

        if born_charges is not None:
            vibrational_data.set_born_charges(born_charges)
        else:
            becs = numpy.array([numpy.eye(3), -1 * numpy.eye(3)])
            vibrational_data.set_born_charges(becs)

        if dph0 is not None:
            vibrational_data.set_raman_tensors(raman_tensors=dph0)
        else:
            dph0 = numpy.zeros((2, 3, 3, 3))
            dph0[0][0][0][0] = +1
            dph0[1][0][0][0] = -1
            vibrational_data.set_raman_tensors(raman_tensors=dph0)

        if nlo is not None:
            vibrational_data.set_nlo_susceptibility(nlo_susceptibility=nlo)
        else:
            vibrational_data.set_nlo_susceptibility(nlo_susceptibility=numpy.zeros((3, 3, 3)))

        if forces is not None:
            vibrational_data.set_forces(forces)
        else:
            vibrational_data.set_forces(sets_of_forces=[[[1, 0, 0], [-1, 0, 0]], [[2, 0, 0], [-2, 0, 0]]])

        return vibrational_data

    return _generate_vibrational_data_from_forces


@pytest.fixture
def generate_phonopy_calcjob_node():
    """Generate an instance of `CalcJobNode`."""

    def _generate_phonopy_calcjob_node(exit_status=0):
        from aiida import orm
        from aiida.common import LinkType
        from plumpy import ProcessState

        node = orm.CalcJobNode().store()
        node.set_process_state(ProcessState.FINISHED)
        node.set_exit_status(exit_status)

        parameters = orm.Dict({'some': 'output'}).store()
        parameters.base.links.add_incoming(node, link_type=LinkType.CREATE, link_label='output_parameters')

        return node

    return _generate_phonopy_calcjob_node
