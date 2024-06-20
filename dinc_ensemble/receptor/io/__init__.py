"""
Module for all receptor loading related stuff.
"""

from .receptor_format_registry import load_receptor

from .receptor_specific_loaders import ReceptorPDBLoader
from .receptor_specific_loaders import ReceptorPDBQTLoader
from .receptor_format_registry import ReceptorFormatRegistry

ReceptorFormatRegistry.register_loader(".pdb", ReceptorPDBLoader)
ReceptorFormatRegistry.register_loader(".pdbqt", ReceptorPDBQTLoader)
