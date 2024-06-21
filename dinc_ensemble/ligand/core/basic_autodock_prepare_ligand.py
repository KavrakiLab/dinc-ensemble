'''
Autodock has a neccessary procedure for preparing ligands - make sure we follow if before loading a DINCMolecule object!

Description of parameters...
     -l     ligand_filename (.pdb or .mol2 or .pdbq format)
Optional parameters:
    verbose                         verbose output
    outputfilename                  (default output filename is ligand_filename_stem + .pdbqt)
    dict                            dictionary to write types list and number of active torsions " 
    repairs                         type(s) of repairs to make:\n\t\t bonds_hydrogens, bonds, hydrogens (default is to do no repairs)
    charges_to_add                  do not add charges (default is to add gasteiger charges)
    preserve_charge_types           preserve input charges on an atom type, eg -p Zn
                                       (defau's not to preserve charges on any specific atom type)
    cleanup                         cleanup type:\n\t\t nphs_lps, nphs, lps, '' (default is 'nphs_lps') 
    allowed_bonds                   type(s) of bonds to allow to rotate 
                                       (defau'ets 'backbone' rotatable and 'amide' + 'guanidinium' non-rotatable)
    root                            index for root
    check_for_fragments             check for and use largest non-bonded fragment (default is not to do this)
    mode                            interactive (default is automatic output)
    bonds_to_inactivate             string of bonds to inactivate composed of 
                                       'ero-based atom indices eg 5_13_2_10  
                                       wi'nactivate atoms[5]-atoms[13] bond 
                                       a'toms[2]-atoms[10] bond 
                                       (defau's not to inactivate any specific bonds)
    inactivate_all_torsions         inactivate all active torsions     
                                       (defau's leave all rotatable active except amide and guanidinium)
    attach_nonbonded_fragments      attach all nonbonded fragments 
    attach_singletons               attach all nonbonded singletons: 
                                       'sets attach all nonbonded fragments too
                                       (defau's not to do this)
    assign_unique_names            assign each ligand atom a unique name: newname is original name plus its index(1-based)
'''

from MolKit.molecule import Molecule as MolKitMolecule
from AutoDockTools.MoleculePreparation import AD4LigandPreparation

from typing import Optional
verbose = False

def prepare_ligand4(
        mol:                        MolKitMolecule,
        verbose:                    Optional[bool] = False,
        outputfilename:             Optional[str] = None,
        dict:                       Optional[str] = None,
        repairs:                    Optional[str] = "bonds_hydrogens",
        charges_to_add:             Optional[str] = "gasteiger",
        preserve_charge_types:      Optional[str] = '',
        cleanup:                    Optional[str] = "nphs_lps",
        # Essentially this is for special bonds that are usually rigid like:
        # Those are: amide, backbone, guanidinium, all
        # Here we can allow them if we want to make an exception
        # Backbone is goof to allow 
        allowed_bonds:              Optional[str] = "backbone", 
        # For now this can stay 
        # We can later tailor this in the fragmentation step
        root:                       Optional[str] = 'auto', 
        check_for_fragments:        Optional[str] = False,
        mode:                       Optional[str]  = 'automatic',
        bonds_to_inactivate:        Optional[str] = "",
        inactivate_all_torsions:    Optional[bool] = False,
        attach_nonbonded_fragments: Optional[bool] = False,
        attach_singletons:          Optional[bool] = False,
        # original default - False
        # DINC important - keep it True!
        assign_unique_names:        Optional[bool] = True,
        build_bonds_by_dist:        Optional[bool] = True
):


    # step 1 - read in all the atoms!
    coord_dict = {}
    for a in mol.allAtoms: coord_dict[a] = a.coords
    
    # step 2 - assign unique names if needed
    if assign_unique_names:  
        assign_new_names(mol.allAtoms)
                
        if verbose:
            print("renamed %d atoms: each newname is the original name of the atom plus its (1-based) uniqIndex" %(len(mol.allAtoms)))
    
    # step 3 - assign bonds by distance (if some bonds are missing?)
    if build_bonds_by_dist:
        mol.buildBondsByDistance()
    
    # step 4 - save charges to perserve them if needed
    if charges_to_add is not None:
        preserved = {}
        preserved_types = preserve_charge_types.split(',') 
        for t in preserved_types:
            if not len(t): continue
            try:
                ats = mol.allAtoms.get(lambda x: x.autodock_element==t)
                for a in ats:
                    if a.chargeSet is not None:
                        preserved[a] = [a.chargeSet, a.charge]
            except AttributeError:
                ats = mol.allAtoms.get(lambda x: x.element==t)
                for a in ats:
                    if a.chargeSet is not None:
                        preserved[a] = [a.chargeSet, a.charge]
            if verbose:
                print(" preserved = ", end=' ') 
                for key, val in list(preserved.items()):
                    print("key=", key)
                    print("val =", val)
    
    # step 5 - prepare with AD4LigandPreparation
    # 1. optional - attach nonbonded fragments
    # 2. LigandPreparation:
    #      a. managing definition of flexibility pattern in ligand
    #          using metaphor of a Tree
    #           +setting ROOT
    #           +setting pattern of rotatable bonds
    #           +setting TORSDOF, number of possible rotatable bonds-
    #                         number of bonds which only rotate hydrogens
    #          optional:
    #           -disallowing amide torsions
    #           -disallowing peptidebackbone torsions
    #           -disallowing guanidinium torsions
    #           +/- toggling activity of any rotatable bond
    #       b. writing an outputfile with keywords recognized by AutoDock
    #       c. writing types list and number of active torsions to a dictionary file
    # 3. add charge error tolerance (needed for AD4 force field)
    # 4. write in the autudock way

    if verbose:
        print("setting up LPO with mode=", mode, end=' ')
        print("and outputfilename= ", outputfilename)
        print("and check_for_fragments=", check_for_fragments)
        print("and bonds_to_inactivate=", bonds_to_inactivate)

    LPO = AD4LigandPreparation(mol, mode, repairs, charges_to_add, 
                            cleanup, allowed_bonds, root, 
                            outputfilename=outputfilename,
                            dict=dict, check_for_fragments=check_for_fragments,
                            bonds_to_inactivate=bonds_to_inactivate, 
                            inactivate_all_torsions=inactivate_all_torsions,
                            attach_nonbonded_fragments=attach_nonbonded_fragments,
                            attach_singletons=attach_singletons, inmem=True)
    
    # step 6 - restore charges ig AD4Prep changed them
    if charges_to_add is not None:
        #restore any previous charges
        for atom, chargeList in list(preserved.items()):
            atom._charges[chargeList[0]] = chargeList[1]
            atom.chargeSet = chargeList[0]
            if verbose: print("set charge on ", atom.full_name(), " to ", atom.charge)
    
    # step 6 - return codes and exits
    if verbose: print("returning ", mol.returnCode) 
    bad_list = []
    for a in mol.allAtoms:
        if a in list(coord_dict.keys()) and a.coords!=coord_dict[a]: 
            bad_list.append(a)
    if len(bad_list):
        if verbose: print(len(bad_list), ' atom coordinates changed!')    
        for a in bad_list:
            if verbose: print(a.name, ":", coord_dict[a], ' -> ', a.coords)
    else:
        if verbose: print("No change in atomic coordinates")

    # step 7 - reassign unique names if needed [in case of hydrogens]
    if assign_unique_names:  # added to simplify setting up covalent dockings 8/2014
        assign_new_names(mol.allAtoms)
    

    #if mol.returnCode!=0: 
    #    raise ValueError(str(mol.returnMsg))

import re

def has_special_characters(s, pat=re.compile('[@_!#$%^&*()<>?/\|}{~:]')):
    if pat.search(s):
        return True
    else:
        return False
    
def assign_new_names(atoms):
    
    unique_atoms_cnt = {}
    for i, at in enumerate(atoms):
        # make sure at.name does not have special characters, if it does avoid them
        #if has_special_characters(at.name):
        #    raise ValueError("Molecule has atom name with special characters: {}".format(at.name))
        name = at.name
        new_name = None
        if name not in unique_atoms_cnt:
            unique_atoms_cnt[name] = 0
            new_name = name
        else:
            cnt = unique_atoms_cnt[name] 
            new_name = name + str(cnt)
            if len(new_name) <= 3:
                unique_atoms_cnt[name] = cnt+1
            else:
                name_base = new_name[:2]
                name_ext = new_name[2]
                next_ext = name_ext
                new_name = name_base+name_ext
                exists = True
                while exists:
                    next_ext = chr(ord(next_ext)+1)
                    new_name = name_base+next_ext
                    if new_name not in unique_atoms_cnt:
                        exists = False
                        unique_atoms_cnt[new_name] = 1
                
        at.name = new_name
    return atoms
