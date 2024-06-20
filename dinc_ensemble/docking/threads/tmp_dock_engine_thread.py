from os import path
from subprocess import call
from threading import Thread
from abc import ABCMeta, abstractmethod
from pathlib import Path

from ...parameters.core import DincCoreParams
from ...ligand import DINCFragment, DINCMolecule
from ...receptor.core import DINCReceptor
from ..dinc_job_elem import DINCThreadElem

debug = 1
##
# The DINCDockEngineThread class allows running multiple docking jobs in parallel using multi-threading.
#   minimum interface for running a single docking thread
#   extend this class with any engine of your choice
##
class DINCDockEngineThread(Thread, 
                     metaclass = ABCMeta):

    def __init__(self, job_name: str, 
                 ligand: DINCMolecule, 
                 receptor: DINCReceptor,
                 output_dir: Path,
                 fragment_index: int = -1,
                 replica_id: int = 0):
        
        self.job_name = job_name
        self.ligand = ligand
        self.receptor = receptor
        self.output_dir = output_dir
        self.frag_index = fragment_index
        self.replica_id = replica_id
        
        self._result_conformations = None 
        self._result_energies = None 

        if debug: 
            print("-------------------------------------")
            print("DINCEnsemble thread init for: {}".format(self.job_name))
            print("Ligand: "+ path.basename(self.ligand_path))
            print("Receptor: "+ path.basename(self.receptor_path))

        Thread.__init__(self)

    @abstractmethod
    @property
    def conformations(self):
        pass
    
    @abstractmethod
    @property
    def energies(self):
        pass
    
    # Prepare for docking:
    # - define binding box
    # - generate FF maps
    @abstractmethod
    def prepare(self):
        pass

    # Optimize ligand
    @abstractmethod
    def optimize(self):
        pass

    # Optimize ligand
    @abstractmethod
    def score(self):
        pass

    # Randomize ligand
    @abstractmethod
    def randomize(self):
        pass

    # Randomize ligand
    @abstractmethod
    def dock(self):
        pass

    # Run incremental docking
    def randomize_and_dock(self):
        self.randomize()
        self.dock()





