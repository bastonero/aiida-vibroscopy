# -*- coding: utf-8 -*-
"""Tests for the :mod:`workflows.phonons.iraman` module."""
import pytest
import numpy
from aiida import orm


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
        from aiida.common import LinkType

        node = orm.WorkflowNode().store()

        parameters = orm.Dict(dict={'number_of_bands': 5}).store()
        parameters.add_incoming(node, link_type=LinkType.RETURN, link_label='output_parameters')

        remote_folder = orm.RemoteData(computer=fixture_localhost, remote_path='/tmp').store()
        remote_folder.add_incoming(node, link_type=LinkType.RETURN, link_label='remote_folder')
        remote_folder.store()

        return node

    return _generate_base_scf_workchain_node


@pytest.fixture
def generate_workchain_iraman(generate_workchain, generate_structure, generate_inputs_pw_base, generate_inputs_dielectric):
    """Generate an instance of a `IRamanSpectraWorkChain`."""

    def _generate_workchain_iraman(append_inputs=None, return_inputs=False):
        entry_point = 'quantumespresso.vibroscopy.spectra.iraman'

        structure = generate_structure()

        scf_inputs = generate_inputs_pw_base()
        scf_inputs['pw'].pop('structure')

        dielectric_inputs = generate_inputs_dielectric(electric_field_scale=1.0)
        dielectric_inputs['scf']['pw'].pop('structure')
        dielectric_inputs.pop('clean_workdir')

        inputs = {
            'structure': structure,
            'scf': scf_inputs,
            'dielectric_workchain':dielectric_inputs
        }

        if return_inputs:
            return inputs

        if append_inputs is not None:
            inputs.update(append_inputs)

        process = generate_workchain(entry_point, inputs)

        return process

    return _generate_workchain_iraman


def generate_invalid_nac_parameters():
    """Generate invalid nac parameters input for `IRamanSpectraWorkChain`."""
    nac = orm.ArrayData()
    nac.set_array('invalid',numpy.eye(3))

    return nac

def generate_valid_nac_parameters():
    """Generate valid nac parameters input for `IRamanSpectraWorkChain`."""
    nac = orm.ArrayData()
    nac.set_array('dielectric',numpy.eye(3))
    bec = numpy.array([numpy.eye(3), -1*numpy.eye(3)])
    nac.set_array('born_charges', bec)

    return nac


@pytest.mark.usefixtures('aiida_profile')
def test_setup(generate_workchain_iraman):
    """Test `IRamanSpectraWorkChain` setep method."""
    process = generate_workchain_iraman()
    process.setup()

    for key in ('preprocess_data', 'is_magnetic', 'plus_hubbard'):
        assert key in process.ctx

    assert process.ctx.run_parallel == True
    assert process.ctx.plus_hubbard == False


def test_run_forces(generate_workchain_iraman, generate_base_scf_workchain_node):
    """Test `IRamanSpectraWorkChain.run_forces` method."""
    process = generate_workchain_iraman()

    process.setup()
    process.run_base_supercell()

    assert 'scf_supercell_0' in process.ctx

    process.ctx.scf_supercell_0 = generate_base_scf_workchain_node()
    process.run_forces()

    assert 'supercells' in process.outputs
    assert 'supercell_1' in process.outputs['supercells']
    assert 'scf_supercell_1' in process.ctx
