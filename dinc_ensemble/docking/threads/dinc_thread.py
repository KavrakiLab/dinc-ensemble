from os import path
from threading import Thread
from abc import ABCMeta, abstractmethod


from ..dinc_job_elem import DINCThreadElem

import logging
logger = logging.getLogger('dinc_ensemble.docking.run')
logger.setLevel(logging.DEBUG)

##
# The DockingThread class allows running multiple docking jobs in parallel using multi-threading.
#   fragment: name of the fragment to be docked, in the format <ligand_name>_frag_<X>_conf_<Y>
#   receptor: name of the file containing the protein receptor
#   flex: name of the file containing the flexible residues of the protein receptor
#   params: parameters of the docking job
##
class DINCDockThread(Thread, 
                     metaclass = ABCMeta):

    def __init__(self,
                 thread_elem: DINCThreadElem):
        
        self.thread_elem = thread_elem

        self.fragment = thread_elem.data.fragment
        self.ligand = thread_elem.data.ligand
        self.receptor = thread_elem.data.receptor
        
        self.receptor_name = thread_elem.job_info.receptor_name
        self.ligand_name = thread_elem.job_info.ligand_name
        self.fragment_name = self.ligand_name
        if self.fragment is not None:
            self.fragment_name = self.fragment._molecule.molkit_molecule.name

        self.frag_index = thread_elem.data.iterative_step
        self.replica = thread_elem.data.replica

        # TODO: unique counter for threads?

        logger.info("-------------------------------------")
        logger.info("DINCEnsemble thread init for: ".format(self.frag_index))
        logger.info("Fragment: "+ self.fragment_name)
        logger.info("Receptor: "+ path.basename(self.receptor_name))
        logger.info("Replica: {}".format(self.replica))
        
        Thread.__init__(self)

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





