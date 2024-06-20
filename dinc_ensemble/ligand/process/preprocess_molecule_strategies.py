from MolKit.hydrogenBuilder import HydrogenBuilder
from MolKit.chargeCalculator import GasteigerChargeCalculator
from AutoDockTools.atomTypeTools import (
    AutoDock4_AtomTyper,
    LonepairMerger,
    NonpolarHydrogenMerger,
)

from .preprocess_molecule_abstract_strategy import PreprocessMoleculeStrategy
from dinc_ensemble.ligand.core import DINCMolecule
class AddHydrogens(PreprocessMoleculeStrategy):

    strategy_name = "add_hydrogens"
    strategy_description: str = """
        Adds hydrogens to the ligand molecule using the MolKit HydrogenBuilder 
        implementation. 
        
        First, MolKit checks for bonds in the molecule. 
        If there are no bonds, it builds them by distance. 
        It assigns hybridization to the atoms in the molecule and assigns bond orders.
        MolKit relies on Pybel for this calculation.
        Finally all hydrogens are added (polar and nonpolar). 
        """
    
    @classmethod
    def process_specific(cls, ligand: DINCMolecule) -> DINCMolecule:

        try:
            hydrogens_N = HydrogenBuilder().addHydrogens(ligand.molkit_molecule)
            print("Added {} hydrogens to the ligand.".format(hydrogens_N))
        except Exception as e:
            raise RuntimeError("DINC failed to add hydrogens to the molecule.") from e
        return ligand

class MergeLonePairs(PreprocessMoleculeStrategy):

    strategy_name = "merge_lonepairs"
    strategy_description: str = """
    Merge Lone Pair of electron atoms in a molecule using the MolKit LonepairMerger.

    This function checks for bonds in the atoms. If there are no bonds, 
    t builds them by distance.
    It finds atoms that are either 'Lp', 'lp' or 'Xx' with names starting with 'L'. 
    If there are no such atoms, it returns an empty list. 
    For each such atom, it increments the charge of the atom it is bonded to, 
    removes the atom from the molecule and deletes it.
    """

    @classmethod
    def process_specific(cls, ligand: DINCMolecule) -> DINCMolecule:

        try:
            return_lp = LonepairMerger().mergeLPS(ligand.molkit_molecule.allAtoms)
            lp_N = None
            if isinstance(return_lp, list):
                lp_N = len(return_lp)
            elif isinstance(return_lp, int):
                    lp_N = return_lp
            else:
                raise RuntimeError("Error running the lone pair merger")
            print("Merged {} lone pairs of the ligand.".format(lp_N))
        except Exception as e:
            raise RuntimeError("DINC failed to merge lone pairs of the molecule.") from e
        
        return ligand
    
class AddGasteigerCharges(PreprocessMoleculeStrategy):
    
    strategy_name = "add_gasteiger_charges"
    strategy_description: str =  """
    Assignes gasteiger charges to atoms in a molecule using the MolKit 
    GasteigerChargeCalculator.

    This function checks if atoms have 'babel_types'. 
    If not, it assigns hybridization to the atoms.
    It then computes Gasteiger charges for the atoms, 
    adds them to the atoms, and returns a list of the new charges.
    """

    @classmethod
    def process_specific(cls, ligand: DINCMolecule) -> DINCMolecule:

        try:
            charges_result = GasteigerChargeCalculator().addCharges(ligand.molkit_molecule.allAtoms)
            print("Added {} charges to the ligand atoms.".format(len(charges_result)))
            print("Charges: {}".format(charges_result))
        except Exception as e:
            raise RuntimeError("DINC failed to add Gasteiger charges to the molecule.") from e
        return ligand

class MergeNonpolarHydrogens(PreprocessMoleculeStrategy):

    strategy_name = "merge_nonpolar_hydrogens"
    strategy_description: str = """
    Merges nonpolar hydrogens in a molecule using the MolKit 
    NonpolarHydrogenMerger.

    This function checks for bonds in the atoms. 
    If there are no bonds, it builds them by distance.
    It finds atoms that are Hydrogens ('H') and are bonded to Carbon atoms ('C'). 
    For each such atom, it increments the charge of the Carbon atom it is bonded to, 
    removes the Hydrogen atom from the molecule and deletes it. 
    Finally, it renumbers the atoms in the molecule.
    """
    @classmethod
    def process_specific(cls, ligand: DINCMolecule) -> DINCMolecule:

        try:
            nonpolar_h_N = NonpolarHydrogenMerger().mergeNPHS(ligand.molkit_molecule.allAtoms)
            print("Merged {} nonpolar hydrogens of the ligand.".format(nonpolar_h_N))
        except Exception as e:
            raise RuntimeError("DINC failed to add Gasteiger charges to the molecule.") from e
        return ligand
    
class AddAutodock4Types(PreprocessMoleculeStrategy):

    strategy_name = "add_autodock_types"
    strategy_description: str = """
    Sets AutoDock atom types for each atom in the molecule using AutoDock4_AtomTyper.

    This function assigns hybridization to the atoms and sets the AutoDock element 
    for each atom based on its element and babel_type.
    """
    @classmethod
    def process_specific(cls, ligand: DINCMolecule) -> DINCMolecule:

        try:
            AutoDock4_AtomTyper().setAutoDockElements(ligand.molkit_molecule, reassign=True)
        except Exception as e:
            raise RuntimeError("DINC failed to assign AutoDock atom types.") from e
        
        return ligand