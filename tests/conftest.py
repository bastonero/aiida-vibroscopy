# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name
"""Initialise a text database and profile for pytest."""
import collections
import os
import shutil
import tempfile

from aiida.manage.fixtures import fixture_manager
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


@pytest.fixture(scope='session')
def fixture_environment():
    """Set up a complete AiiDA test environment, with configuration, profile, database and repository."""
    with fixture_manager() as manager:
        yield manager


@pytest.fixture(scope='session')
def fixture_work_directory():
    """Return a temporary folder that can be used as for example a computer's work directory."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


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
        from aiida.orm import Code
        return Code(input_plugin_name=entry_point_name, remote_computer_exec=[fixture_localhost, '/bin/true'])

    return _fixture_code


@pytest.fixture(scope='function')
def fixture_database(fixture_environment):
    """Clear the database after each test."""
    yield
    fixture_environment.reset_db()


@pytest.fixture
def generate_calc_job():
    """Fixture to construct a new `CalcJob` instance and call `prepare_for_submission` for testing `CalcJob` classes.

    The fixture will return the `CalcInfo` returned by `prepare_for_submission` and the temporary folder that was
    passed to it, into which the raw input files will have been written.
    """

    def _generate_calc_job(folder, entry_point_name, inputs=None):
        """Fixture to generate a mock `CalcInfo` for testing calculation jobs."""
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager
        from aiida.plugins import CalculationFactory

        manager = get_manager()
        runner = manager.get_runner()

        process_class = CalculationFactory(entry_point_name)
        process = instantiate_process(runner, process_class, **inputs)

        calc_info = process.prepare_for_submission(folder)

        return calc_info

    return _generate_calc_job


@pytest.fixture
def generate_calc_job_node(fixture_localhost):
    """Fixture to generate a mock `CalcJobNode` for testing parsers."""

    def flatten_inputs(inputs, prefix=''):
        """Flatten inputs recursively like :meth:`aiida.engine.processes.process::Process._flatten_inputs`."""
        flat_inputs = []
        for key, value in inputs.items():
            if isinstance(value, collections.Mapping):
                flat_inputs.extend(flatten_inputs(value, prefix=prefix + key + '__'))
            else:
                flat_inputs.append((prefix + key, value))
        return flat_inputs

    def _generate_calc_job_node(entry_point_name, computer=None, test_name=None, inputs=None, attributes=None):
        """Fixture to generate a mock `CalcJobNode` for testing parsers.

        :param entry_point_name: entry point name of the calculation class
        :param computer: a `Computer` instance
        :param test_name: relative path of directory with test output files in the `fixtures/{entry_point_name}` folder
        :param inputs: any optional nodes to add as input links to the corrent CalcJobNode
        :param attributes: any optional attributes to set on the node
        :return: `CalcJobNode` instance with an attached `FolderData` as the `retrieved` node
        """
        from aiida import orm
        from aiida.common import LinkType
        from aiida.plugins.entry_point import format_entry_point_string

        if computer is None:
            computer = fixture_localhost

        entry_point = format_entry_point_string('aiida.calculations', entry_point_name)

        node = orm.CalcJobNode(computer=computer, process_type=entry_point)
        node.set_attribute('input_filename', 'aiida.in')
        node.set_attribute('output_filename', 'aiida.out')
        node.set_attribute('error_filename', 'aiida.err')
        node.set_option('resources', {'num_machines': 1, 'num_mpiprocs_per_machine': 1})
        node.set_option('max_wallclock_seconds', 1800)

        if attributes:
            node.set_attribute_many(attributes)

        if inputs:
            metadata = inputs.pop('metadata', {})
            options = metadata.get('options', {})

            for name, option in options.items():
                node.set_option(name, option)

            for link_label, input_node in flatten_inputs(inputs):
                input_node.store()
                node.add_incoming(input_node, link_type=LinkType.INPUT_CALC, link_label=link_label)

        node.store()

        retrieved = orm.FolderData()

        if test_name is not None:
            basepath = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(
                basepath, 'parsers', 'fixtures', entry_point_name[len('quantumespresso.'):], test_name
            )
            retrieved.put_object_from_tree(filepath)

        retrieved.add_incoming(node, link_type=LinkType.CREATE, link_label='retrieved')
        retrieved.store()

        remote_folder = orm.RemoteData(computer=computer, remote_path='/tmp')
        remote_folder.add_incoming(node, link_type=LinkType.CREATE, link_label='remote_folder')
        remote_folder.store()

        return node

    return _generate_calc_job_node


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
def generate_code_localhost():
    """Return a `Code` instance configured to run calculations of given entry point on localhost `Computer`."""

    def _generate_code_localhost(entry_point_name, computer):
        from aiida.orm import Code
        plugin_name = entry_point_name
        remote_computer_exec = [computer, '/bin/true']
        return Code(input_plugin_name=plugin_name, remote_computer_exec=remote_computer_exec)

    return _generate_code_localhost


@pytest.fixture
def serialize_builder():
    """Serialize the given process builder into a dictionary with nodes turned into their value representation.

    :param builder: the process builder to serialize
    :return: dictionary
    """

    def serialize_data(data):
        # pylint: disable=too-many-return-statements
        from aiida.orm import BaseType, Code, Dict
        from aiida.plugins import DataFactory

        StructureData = DataFactory('structure')
        UpfData = DataFactory('pseudo.upf')

        if isinstance(data, dict):
            return {key: serialize_data(value) for key, value in data.items()}

        if isinstance(data, BaseType):
            return data.value

        if isinstance(data, Code):
            return data.full_label

        if isinstance(data, Dict):
            return data.get_dict()

        if isinstance(data, StructureData):
            return data.get_formula()

        if isinstance(data, UpfData):
            return f'{data.element}<md5={data.md5}>'

        return data

    def _serialize_builder(builder):
        return serialize_data(builder._inputs(prune=True))  # pylint: disable=protected-access

    return _serialize_builder


@pytest.fixture
def generate_structure():
    """Return a `StructureData` representing bulk silicon."""

    def _generate_structure(sites=None, structure_id=None):
        """Return a `StructureData` representing bulk silicon."""
        from aiida.orm import StructureData

        if sites is None:
            cell = [[3.9625313477, -3.9625313477, 0.0], [-3.9625313477, 0.0, 3.9625313477],
                    [0.0, -3.9625313477, -3.9625313477]]
            structure = StructureData(cell=cell)
        else:
            structure = StructureData()

        if sites is None:
            structure.append_atom(position=(0., 0., 0.), symbols='Mg', name='Mg')
            structure.append_atom(position=(1.98126567385, 1.98126567385, 1.98126567385), symbols='O', name='O')
        else:
            for kind, symbol in sites:
                structure.append_atom(position=(0., 0., 0.), symbols=symbol, name=kind)

        if structure_id == 'silicon':
            param = 5.43
            cell = [[param / 2., param / 2., 0], [param / 2., 0, param / 2.], [0, param / 2., param / 2.]]
            structure = StructureData(cell=cell)
            structure.append_atom(position=(0., 0., 0.), symbols='Si', name='Si')
            structure.append_atom(position=(param / 4., param / 4., param / 4.), symbols='Si', name='Si')

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
def generate_parser():
    """Fixture to load a parser class for testing parsers."""

    def _generate_parser(entry_point_name):
        """Fixture to load a parser class for testing parsers.

        :param entry_point_name: entry point name of the parser class
        :return: the `Parser` sub class
        """
        from aiida.plugins import ParserFactory
        return ParserFactory(entry_point_name)

    return _generate_parser


@pytest.fixture
def generate_inputs_pw(fixture_code, generate_structure, generate_kpoints_mesh, generate_upf_data):
    """Generate default inputs for a `PwCalculation."""

    def _generate_inputs_pw(parameters=None, structure=None):
        """Generate default inputs for a `PwCalculation."""
        from aiida.orm import Dict
        from aiida_quantumespresso.utils.resources import get_default_options

        parameters_base = {
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
        }

        if parameters is not None:
            parameters_base.update(parameters)

        inputs = {
            'code': fixture_code('quantumespresso.pw'),
            'structure': structure or generate_structure(),
            'kpoints': generate_kpoints_mesh(2),
            'parameters': Dict(dict=parameters_base),
            'pseudos': {kind: generate_upf_data(kind) for kind in structure.get_kind_names()},
            'metadata': {
                'options': get_default_options()
            }
        }

        return inputs

    return _generate_inputs_pw


@pytest.fixture
def generate_inputs_dielectric(generate_inputs_pw, generate_structure):
    """Generate default inputs for a `DielectricWorkChain`."""

    def _generate_inputs_dielectric(
        property='raman',
        clean_workdir=True,
        electric_field=None,
        electric_field_scale=None,
        accuracy=None,
        diagonal_scale=None
    ):
        """Generate default inputs for a `DielectricWorkChain`."""
        from aiida.orm import Bool, Float, Int

        structure = generate_structure()
        inputs_scf = generate_inputs_pw(structure=structure)

        kpoints = inputs_scf.pop('kpoints')

        inputs = {
            'property': property,
            'scf': {
                'pw': inputs_scf,
                'kpoints': kpoints,
            },
            'clean_workdir': Bool(clean_workdir),
            'options': {
                'sleep_submission_time': 0
            }
        }

        if electric_field is not None:
            inputs['electric_field'] = Float(electric_field)
        if electric_field_scale is not None:
            inputs['electric_field_scale'] = Float(electric_field_scale)
        if accuracy is not None:
            inputs['central_difference'] = {'accuracy': Int(accuracy)}
        if diagonal_scale is not None:
            inputs['central_difference'] = {'diagonal_scale': Float(diagonal_scale)}

        return inputs

    return _generate_inputs_dielectric


@pytest.fixture
def generate_inputs_second_derivatives(generate_trajectory):
    """Generate default inputs for a `SecondOrderDerivativesWorkChain`."""

    def _generate_inputs_second_derivatives(trial=None, volume=None, elfield=None):
        """Generate default inputs for a `SecondOrderDerivativesWorkChain`."""
        from aiida.orm import Float

        if trial is None:
            data = {
                'null': generate_trajectory(),
                'field0': generate_trajectory(),
                'field1': generate_trajectory(),
                'field2': generate_trajectory(),
            }
        elif trial == 0:
            data = {
                'null': generate_trajectory(),
                'field1': generate_trajectory(),
            }
        elif trial == 1:
            data = {
                'field1': generate_trajectory(),
            }
        elif trial == 2:
            data = {
                'null': generate_trajectory(),
            }

        inputs = {
            'data': data,
        }

        if not volume is None:
            inputs['volume'] = Float(volume)
        else:
            inputs['volume'] = Float(1.0)
        if not elfield is None:
            inputs['elfield'] = Float(elfield)
        else:
            inputs['elfield'] = Float(0.001)

        return inputs

    return _generate_inputs_second_derivatives


# @pytest.fixture(scope='session')
# def generate_upf_data(tmp_path_factory):
#     """Return a `UpfData` instance for the given element a file for which should exist in `tests/fixtures/pseudos`."""

#     def _generate_upf_data(element):
#         """Return `UpfData` node."""
#         from aiida.orm import UpfData

#         with open(tmp_path_factory.mktemp('pseudos') / f'{element}.upf', 'w+b') as handle:
#             handle.write(f'<UPF version="2.0.1"><PP_HEADER element="{element}"/></UPF>'.encode('utf-8'))
#             handle.flush()
#             return UpfData(file=handle.name)

#     return _generate_upf_data


@pytest.fixture(scope='session')
def generate_upf_data():
    """Return a `UpfData` instance for the given element a file for which should exist in `tests/fixtures/pseudos`."""

    def _generate_upf_data(element):
        """Return `UpfData` node."""
        import io

        from aiida_pseudo.data.pseudo import UpfData
        content = f'<UPF version="2.0.1"><PP_HEADER\nelement="{element}"\nz_valence="4.0"\n/></UPF>\n'
        stream = io.BytesIO(content.encode('utf-8'))
        return UpfData(stream, filename=f'{element}.upf')

    return _generate_upf_data


@pytest.fixture(scope='session')
def generate_upf_family(generate_upf_data):
    """Return a `UpfFamily` that serves as a pseudo family."""

    def _generate_upf_family(structure, label='SSSP-testing2'):
        from aiida.common import exceptions
        from aiida.orm import UpfFamily

        try:
            existing = UpfFamily.objects.get(label=label)
        except exceptions.NotExistent:
            pass
        else:
            UpfFamily.objects.delete(existing.pk)

        family = UpfFamily(label=label)

        pseudos = {}

        for kind in structure.kinds:
            pseudo = generate_upf_data(kind.symbol).store()
            pseudos[pseudo.element] = pseudo

        family.store()
        family.add_nodes(list(pseudos.values()))

        return family

    return _generate_upf_family


@pytest.fixture(scope='session', autouse=True)
def sssp(aiida_profile, generate_upf_data):
    """Create an SSSP pseudo potential family from scratch."""
    from aiida.common.constants import elements
    from aiida.plugins import GroupFactory

    # aiida_profile.clear_profile()

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
            filename = os.path.join(dirpath, f'{element}.upf')

            with open(filename, 'w+b') as handle:
                with upf.open(mode='rb') as source:
                    handle.write(source.read())
                    handle.flush()

            cutoffs[element] = {
                'cutoff_wfc': 30.0,
                'cutoff_rho': 240.0,
            }

        label = 'SSSP/1.1/PBE/efficiency'
        family = SsspFamily.create_from_folder(dirpath, label)

    family.set_cutoffs(cutoffs, stringency, unit='Ry')

    return family


@pytest.fixture
def generate_inputs_pw_base(generate_inputs_pw, generate_structure):
    """Generate default inputs for a `PwBaseWorkChain`."""

    def _generate_inputs_pw_base():
        """Generate default inputs for a `PwBaseWorkChain`."""
        structure = generate_structure()
        inputs_scf = generate_inputs_pw(structure=structure)

        kpoints = inputs_scf.pop('kpoints')

        inputs = {
            'pw': inputs_scf,
            'kpoints': kpoints,
        }

        return inputs

    return _generate_inputs_pw_base


@pytest.fixture
def generate_base_scf_workchain_node(fixture_localhost):
    """Generate an instance of `WorkflowNode`."""

    def _generate_base_scf_workchain_node():
        from aiida import orm
        from aiida.common import LinkType
        from aiida.plugins.entry_point import format_entry_point_string

        computer = fixture_localhost
        entry_point_name = 'quantumespresso.pw'

        entry_point = format_entry_point_string('aiida.calculations', entry_point_name)
        calcjob_node = orm.CalcJobNode(computer=computer, process_type=entry_point)
        calcjob_node.store()

        node = orm.WorkflowNode().store()

        parameters = orm.Dict(dict={'number_of_bands': 5}).store()
        parameters.add_incoming(node, link_type=LinkType.RETURN, link_label='output_parameters')

        remote_folder = orm.RemoteData(computer=computer, remote_path='/tmp').store()
        remote_folder.add_incoming(node, link_type=LinkType.RETURN, link_label='remote_folder')
        remote_folder.add_incoming(calcjob_node, link_type=LinkType.CREATE, link_label='remote_folder')
        remote_folder.store()

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
                tensors.set_array('raman_susceptibility', dph0_array)

                nlo_array = numpy.zeros((3, 3, 3))
                tensors.set_array('nlo_susceptibility', nlo_array)

            tensors.store()

            tensors.add_incoming(node, link_type=LinkType.RETURN, link_label=f'tensors__{label}')

        return node

    return _generate_dielectric_workchain_node


@pytest.fixture
def generate_vibrational_data(generate_structure):
    """Generate a `VibrationalFrozenPhononData`."""

    def _generate_vibrational_data(dielectric=None, born_charges=None, dph0=None, nlo=None, forces=None):
        """Return a `VibrationalFrozenPhononData` with bulk silicon as structure."""
        from aiida.plugins import DataFactory
        import numpy

        # from aiida_vibroscopy.data.vibro_fp import VibrationalFrozenPhononData

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
            vibrational_data.set_raman_susceptibility(raman_susceptibility=dph0)
        else:
            dph0 = numpy.zeros((2, 3, 3, 3))
            dph0[0][0][0][0] = +1
            dph0[1][0][0][0] = -1
            vibrational_data.set_raman_susceptibility(raman_susceptibility=dph0)

        if nlo is not None:
            vibrational_data.set_nlo_susceptibility(nlo_susceptibility=nlo)
        else:
            vibrational_data.set_nlo_susceptibility(nlo_susceptibility=numpy.zeros((3, 3, 3)))

        if forces is not None:
            vibrational_data.set_forces(forces)
        else:
            vibrational_data.set_forces([[[1, 0, 0], [-1, 0, 0]], [[2, 0, 0], [-2, 0, 0]]])

        return vibrational_data

    return _generate_vibrational_data
