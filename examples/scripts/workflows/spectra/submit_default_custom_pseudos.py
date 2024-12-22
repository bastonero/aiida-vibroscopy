# -*- coding: utf-8 -*-
# pylint: disable=line-too-long,wildcard-import,pointless-string-statement,unused-wildcard-import
"""Submit an IRamanSpectraWorkChain via the get_builder_from_protocol with custom pseudo potentials."""
from aiida import load_profile
from aiida.engine import submit
from aiida.orm import *
from aiida_quantumespresso.common.types import ElectronicType

from aiida_vibroscopy.workflows.spectra.iraman import IRamanSpectraWorkChain

load_profile()

# =================== HOW TO STORE CUSTOM PSEUDOS ====================== #
# Please consult also aiida-pseudo documentation.
# Prepare a folder with the pseudopotentials you want to use, with all the elements.
# The format should be ELEMENT.EVENTUAL_DETAILS.UPF . For instance:
#   * Si.upf
#   * Si.UPF
#   * Si.us-v1.0.upf
#   * Si.paw-rjjk.v1.3.upf
# Please prepare a folder like:
# -- MyPseudos
# -- |_ Si.upf
# -- |_ O.upf
# -- |_ ...
# Then run
# $ aiida-pseudo install family MyPseudos LABEL -P pseudo.upf
# Substitute LABEL with some significant label referring to the pseudo family you use.
# For instance, good practices:
#   * LDA/NC/1.1
#   * PseudoDojo/LDA/US/standard/1.1
# Consider that you can also install directly well tested pseudo potentials,
# for example from the SSSP library, with the following:
# $ aiida-pseudo install sssp -v 1.3 -x PBEsol -p efficiency
# This will automatically download and store the pseudopotentials in a family.
# Register the name. You can inspect the pseudo potential families you have with
# $ aiida-pseudo list

# =============================== INPUTS =============================== #
# Please, change the following inputs.
pw_code_label = 'pw@localhost'
structure_id = 0  # PK or UUID of your AiiDA StructureData
protocol = 'fast'  # also 'moderate' and 'precise'; 'moderate' should be good enough in general
pseudo_family_name = 'LABEL'  # here the LABEL you registered before, or e.g. SSSP/1.3/PBEsol/efficiency for the SSSP example showed
# ====================================================================== #
# If you don't have a StructureData, but you have a CIF or XYZ, or similar, file
# you can import your structure uncommenting the following:
# from ase.io import read
# atoms = read('/path/to/file.cif')
# structure = StructureData(ase=atoms)
# structure.store()
# structure_id =  structure.pk
# print(f"Your structure has been stored in the database with PK={structure_id}")


def main():
    """Submit an IRamanSpectraWorkChain calculation."""
    code = load_code(pw_code_label)
    structure = load_node(structure_id)
    kwargs = {'electronic_type': ElectronicType.INSULATOR}

    pseudo_family = load_group(pseudo_family_name)
    pseudos = pseudo_family.get_pseudos(structure=structure)

    builder = IRamanSpectraWorkChain.get_builder_from_protocol(
        code=code,
        structure=structure,
        protocol=protocol,
        **kwargs,
    )

    builder.dielectric.scf.pw.pseudos = pseudos
    builder.phonon.scf.pw.pseudos = pseudos

    calc = submit(builder)
    print(f'Submitted IRamanSpectraWorkChain with PK={calc.pk} and UUID={calc.uuid}')
    print('Register *at least* the PK number, e.g. in you submit script.')
    print('You can monitor the status of your calculation with the following commands:')
    print('  * verdi process status PK')
    print('  * verdi process list -L IRamanSpectraWorkChain # list all running IRamanSpectraWorkChain')
    print(
        '  * verdi process list -ap1 -L IRamanSpectraWorkChain # list all IRamanSpectraWorkChain submitted in the previous 1 day'
    )
    print('If the WorkChain finishes with exit code 0, then you can inspect the outputs and post-process the data.')
    print('Use the command')
    print('  * verdi process show PK')
    print('To show further information about your WorkChain. When finished, you should see some outputs.')
    print('The main output can be accessed via `load_node(PK).outputs.vibrational_data.numerical_accuracy_*`.')
    print('You have to complete the remaning `*`, which depends upond the accuracy of the calculation.')
    print('See also the documentation and the reference paper for further details.')


if __name__ == '__main__':
    """Run script."""
    main()
