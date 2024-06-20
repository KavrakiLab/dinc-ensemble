import logging
logger = logging.getLogger(__name__)

from .ligand_io_format_registry import LigandIOFormatRegistry

logger.info("Loading ligands without Meeko!")
from .ligand_specific_loaders import LigandMol2Loader
from .ligand_specific_loaders import LigandPDBLoader
from .ligand_specific_loaders import LigandFastaLoader
from .ligand_specific_loaders import LigandSDFLoader

from .ligand_io_format_registry import load_ligand
from .ligand_io_format_registry import write_ligand

LigandIOFormatRegistry.register_loader(".mol2", LigandMol2Loader)
LigandIOFormatRegistry.register_loader(".pdb", LigandPDBLoader)
LigandIOFormatRegistry.register_loader(".fasta", LigandFastaLoader)
LigandIOFormatRegistry.register_loader(".sdf", LigandSDFLoader)

from .ligand_specific_writers import LigandPDBWriter
from .ligand_specific_writers import LigandMol2Writer
from .ligand_specific_writers import LigandSDFWriter

LigandIOFormatRegistry.register_writer(".pdb", LigandPDBWriter)
LigandIOFormatRegistry.register_writer(".mol2", LigandMol2Writer)
LigandIOFormatRegistry.register_writer(".sdf", LigandSDFWriter)