'''
Based on the ADFR Suite's preparation of receptor.
Implementation allows for running without invoking cmd line script.
'''
from __future__ import annotations
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .receptor import DINCReceptor


from MolKit import Read
import MolKit.molecule
import MolKit.protein
from AutoDockTools.MoleculePreparation import AD4ReceptorPreparation


import logging
logger = logging.getLogger('dinc_ensemble.receptor')


def prepare_receptor_ad4(
        receptor: DINCReceptor,
        verbose: bool = False,
        repairs: str = "checkhydrogens",
        charges_to_add: str = "gasteiger",
        preserve_charge_types: str = None,
        # delete alternative positions, also important!!
        cleanup: str = "nphs_lps_waters_nonstdres_deleteAltB",
        # remove hetatms!! importand addition!
        delete_single_nonstd_residues: bool = True,
        output_dict = None,
        unique_atom_names: bool = False
):
    """
    prepare_receptor_ad4

    This function prepares a receptor for docking simulations by applying various repair options,
    calculating charges, and cleaning up the molecule based on user-specified parameters.

    Input
    ----------
    receptor : DINCReceptor
        DINCReceptor object
    verbose : bool, optional
        Enable verbose output (default is minimal output).
    repairs : {'bonds_hydrogens', 'bonds', 'hydrogens', 'checkhydrogens', None}, optional
        Type(s) of repairs to make:
        - 'bonds_hydrogens': Build bonds and add hydrogens.
        - 'bonds': Build a single bond from each atom with no bonds to its closest neighbor.
        - 'hydrogens': Add hydrogens.
        - 'checkhydrogens': Add hydrogens only if there are none already.
        - 'None': Do not make any repairs (default).
        (default is 'checkhydrogens')
    charges_to_add : bool, optional
        Preserve all input charges; do not add new charges (default is addition of Gasteiger charges).
    preserve_charge_types : str, optional
        Preserve input charges on specific atom types, e.g., -p Zn -p Fe.
    clanup : {'nphs', 'lps', 'waters', 'nonstdres', 'deleteAltB'}, optional
        Cleanup type:
        - 'nphs': Merge charges and remove non-polar hydrogens.
        - 'lps': Merge charges and remove lone pairs.
        - 'waters': Remove water residues.
        - 'nonstdres': Remove chains composed entirely of residues of types other than the standard 20 amino acids.
        - 'deleteAltB': Remove XX@B atoms and rename XX@A atoms -> XX (default is 'nphs_lps_waters_nonstdres').
        default: nphs_lps_waters_nonstdres
    delete_single_nonstd_residues : bool, optional
        Delete every nonstandard residue from any chain (default is False, meaning not to do this).
    output_dict : str, optional
        File to contain receptor summary information.
    unique_atom_names : bool, optional
        Assign each receptor atom a unique name: new name is original name plus its index (1-based).
        default: False

    Returns
    -------
    None

    Examples
    --------
    >>> from dinc_ensemble import prepare_receptor_ad4
    >>> from dinc_ensemble import load_receptor
    >>> dinc_receptor = load_receptor("/path/to/receptor.pdb")
    >>> prepare_receptor_ad4.prepare_receptor_ad4(dinc_receptor)
    """

    # initialize required parameters
    
    mol = receptor.molkit_receptor
    if unique_atom_names:  # added to simplify setting up covalent dockings 8/2014
        for at in mol.allAtoms:
            if mol.allAtoms.get(at.name) >1:
                at.name = at.name + str(at._uniqIndex +1)
        logger.info("renamed %d atoms: each newname is the original name of the atom plus its (1-based) uniqIndex" %(len(mol.allAtoms)))        
    preserved = {}
    has_autodock_element = False
    if charges_to_add is not None and preserve_charge_types is not None:

        if preserve_charge_types is not None and has_autodock_element==False:
            logger.info('prepare_receptor4: input format does not have autodock_element SO unable to preserve charges on ' + preserve_charge_types)
            logger.info('exiting...')
            return
        preserved_types = preserve_charge_types.split(',') 
        logger.info("preserved_types=", preserved_types)
        for t in preserved_types:
            logger.info('preserving charges on type->', t)
            if not len(t): continue
            ats = mol.allAtoms.get(lambda x: x.autodock_element==t)
            logger.info("preserving charges on ", ats.name)
            for a in ats:
                if a.chargeSet is not None:
                    preserved[a] = [a.chargeSet, a.charge]

    mol.buildBondsByDistance()
    alt_loc_ats = mol.allAtoms.get(lambda x: "@" in x.name)
    len_alt_loc_ats = len(alt_loc_ats)
    if len_alt_loc_ats:
        logger.warning("WARNING!", mol.name, "has",len_alt_loc_ats, ' alternate location atoms!\nUse prepare_pdb_split_alt_confs.py to create pdb files containing a single conformation.\n')

    mode = "automatic"
    outputfilename =""
    inmem = True

    logger.info("setting up RPO with mode=", mode)
    logger.info("and outputfilename= ", outputfilename)
    logger.info("charges_to_add=", charges_to_add)
    logger.info("inmem=", inmem)
    logger.info("delete_single_nonstd_residues=", delete_single_nonstd_residues)

    RPO = AD4ReceptorPreparation(mol, mode, repairs, charges_to_add, 
                        cleanup, outputfilename=outputfilename,
                        preserved=preserved, 
                        delete_single_nonstd_residues=delete_single_nonstd_residues,
                        dict=output_dict, inmem=True, debug=False)    

    if charges_to_add is not None:
        #restore any previous charges
        for atom, chargeList in list(preserved.items()):
            atom._charges[chargeList[0]] = chargeList[1]
            atom.chargeSet = chargeList[0]
