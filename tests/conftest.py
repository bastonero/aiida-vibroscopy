# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name
"""Initialise a text database and profile for pytest."""
import collections
import io
import os
import shutil
import tempfile

import pytest
from aiida.manage.fixtures import fixture_manager

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
def generate_structure():
    """Return a `StructureData` representing bulk silicon."""

    def _generate_structure(sites=None):
        """Return a `StructureData` representing bulk silicon."""
        from aiida.orm import StructureData
            
        cell = [[3.9625313477, -3.9625313477, 0.0],
                [-3.9625313477, 0.0, 3.9625313477],
                [0.0, -3.9625313477, -3.9625313477]]
        #cell = [[1., 1., 0], [1., 0, 1.], [0, 1., 1.]]
        
        structure = StructureData(cell=cell)
        if sites is None:
            structure.append_atom(position=(0., 0., 0.), symbols='Mg', name='Mg')
            structure.append_atom(position=(1.98126567385, 1.98126567385, 1.98126567385), symbols='O', name='O')
        else:
            for kind, symbol in sites:
                structure.append_atom(position=(0., 0., 0.), symbols=symbol, name=kind)

        return structure

    return _generate_structure


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
def generate_elfield():
    """Return a `Float` node."""

    def _generate_elfield(elfield):
        """Return a `Float` with the value of `elfield`."""
        from aiida.orm import Float

        elfield_node = Float(elfield)

        return elfield_node

    return _generate_elfield

@pytest.fixture
def generate_nberrycyc():
    """Return a `Float` node."""

    def _generate_nberrycyc(nberrycyc):
        """Return a `Int` with the value of `nberrycyc`."""
        from aiida.orm import Int

        nberrycyc_node = Int(nberrycyc)

        return nberrycyc_node

    return _generate_nberrycyc

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
def generate_inputs_pw(fixture_code, generate_structure, generate_kpoints_mesh, generate_upf_family):
    """Generate default inputs for a `PwCalculation."""

    def _generate_inputs_pw(parameters=None, structure=None):
        """Generate default inputs for a `PwCalculation."""
        from aiida.orm import Dict
        from aiida.orm.nodes.data.upf import get_pseudos_from_structure
        from aiida_quantumespresso.utils.resources import get_default_options

        parameters_base = {'CONTROL': {'calculation': 'scf'}, 'SYSTEM': {'ecutrho': 400.0, 'ecutwfc': 50.0}}

        if parameters is not None:
            parameters_base.update(parameters)

        inputs = {
            'code': fixture_code('quantumespresso.pw'),
            'structure': structure or generate_structure(),
            'kpoints': generate_kpoints_mesh(2),
            'parameters': Dict(dict=parameters_base),
            'metadata': {
                'options': get_default_options()
            }
        }

        family = generate_upf_family(inputs['structure'])
        inputs['pseudos'] = get_pseudos_from_structure(inputs['structure'], family.label)

        return inputs

    return _generate_inputs_pw


@pytest.fixture
def generate_inputs_finite_electric_fields(generate_inputs_pw, generate_structure, generate_elfield, generate_nberrycyc):
    """Generate default inputs for a `FiniteElectricFieldsWorkChain`."""

    def _generate_inputs_finite_electric_fields(elfield=None, nberrycyc=None, selected_elfield=None):
        """Generate default inputs for a `FiniteElectricFieldsWorkChain`."""
        from aiida.orm import Dict, Float

        structure = generate_structure()
        inputs_elfield_scf = generate_inputs_pw(structure=structure)

        kpoints = inputs_elfield_scf.pop('kpoints')

        elfield_node = elfield or generate_elfield(0.001)
        nberrycyc_node = nberrycyc or generate_nberrycyc(3)
        
        inputs = {
            'elfield_scf': {
                'pw': inputs_elfield_scf,
                'kpoints': kpoints,
            },
            'elfield':elfield_node,
            'nberrycyc':nberrycyc_node,
        }
        
        if not selected_elfield==None:
            from aiida.orm import Int
            inputs['selected_elfield']=Int(selected_elfield)
        
        return inputs

    return _generate_inputs_finite_electric_fields


@pytest.fixture(scope='session')
def generate_upf_data(tmp_path_factory):
    """Return a `UpfData` instance for the given element a file for which should exist in `tests/fixtures/pseudos`."""

    def _generate_upf_data(element):
        """Return `UpfData` node."""
        from aiida.orm import UpfData

        with open(tmp_path_factory.mktemp('pseudos') / f'{element}.upf', 'w+b') as handle:
            handle.write(f'<UPF version="2.0.1"><PP_HEADER element="{element}"/></UPF>'.encode('utf-8'))
            handle.flush()
            return UpfData(file=handle.name)

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
