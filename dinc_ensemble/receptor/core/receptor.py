
from MolKit.protein import Protein as MolKitProtein
from dataclasses import dataclass
from .utils_prepare_ad4 import prepare_receptor_ad4
from pathlib import Path
@dataclass
class DINCReceptor:

    """
    A class representing a receptor in the DINC format.

    Attributes
    ----------
    molkit_receptor : MolKitProtein
        The original MolKitProtein object.
    """
    molkit_receptor: MolKitProtein
    _fname: str

    def __init__(self, 
                 molkit_rec: MolKitProtein,
                 fname: str):
        self.molkit_receptor = molkit_rec
        self._fname = fname
        self.name = Path(fname).stem
        prepare_receptor_ad4(self)
        self.pdbqt_str = self.molkit_receptor.pdbqt_str



 