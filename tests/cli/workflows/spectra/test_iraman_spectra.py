# -*- coding: utf-8 -*-
"""Tests for the ``launch iraman-spectra`` command."""
from pathlib import Path

from aiida_vibroscopy.cli.workflows.spectra.iraman import launch_workflow


# yapf: disable
def test_command_iraman_spectra(run_cli_process_launch_command, fixture_code, filepath_cli_fixture):
    """Test invoking the launch command with only required inputs."""
    code = fixture_code('quantumespresso.pw').store()
    options = [
        '--pw', code.full_label,
        '-o', str(Path(filepath_cli_fixture, 'overrides', 'iraman-spectra.yaml')),
        '-p', 'fast',
        '-F', 'SSSP/1.3/PBEsol/efficiency',
    ]
    run_cli_process_launch_command(launch_workflow, options=options)

    options = [
        '--pw', code.full_label,
        '-p', 'fast',
        '-F', 'SSSP/1.3/PBEsol/efficiency',
    ]
    run_cli_process_launch_command(launch_workflow, options=options)

    options = [
        '--pw', code.full_label,
        '-p', 'fast',
        '-o', str(Path(filepath_cli_fixture, 'overrides', 'iraman-spectra.yaml')),
        '-k', '2', '2', '2', '0.5', '0.5', '0.5',
    ]
    run_cli_process_launch_command(launch_workflow, options=options)

    options = [
        '--pw', code.full_label,
        '-p', 'fast',
        '-k', '2', '2', '2', '0.5', '0.5', '0.5',
    ]
    run_cli_process_launch_command(launch_workflow, options=options)
