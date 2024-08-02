from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import pandas as pd
from ..ligand import DINCMolecule, DINCFragment
from ..receptor import DINCReceptor
from ..parameters import DINC_CORE_PARAMS, \
                            DINC_JOB_TYPE

from shutil import copy as shcopy
from os import path



@dataclass
class DINCJobInfo:
    # DINC Job: single DINC job
    # 1 ligand, 1 receptor

    job_name: str 
    job_root: Path
    ligand: Path
    receptor: Path

    def __post_init__(self):
        # initialize the single job files
        self.job_root.mkdir(parents=True, exist_ok=True)
        new_lig = self.job_root / path.basename(self.ligand)
        shcopy(self.ligand, new_lig)
        self.ligand = new_lig

        old_rec = self.receptor
        new_rec = self.job_root / path.basename(old_rec)
        shcopy(old_rec, new_rec)
        self.receptor = old_rec

    @property
    def ligand_name(self):
        return self.ligand.stem.split(".")[0]
    
    @property
    def receptor_name(self):
        return self.receptor.stem.split(".")[0]

@dataclass
class DINCRunInfo:
    # DINC Run: one full run of DINC
    # 1 ligand, N receptors
    root: Path
    ligand: Path
    receptors: List[Path]
    ensemble: Path
    analysis: Path

    def create_jobs(self) -> List[DINCJobInfo]:

        jobs = []
        for receptor in self.receptors:
            receptor_name = receptor.stem
            ligand_name = self.ligand.stem
            job_name = "job_{}_{}".format(receptor_name, ligand_name)
            job_root = self.root / Path(job_name)
            job = DINCJobInfo(job_name, 
                        job_root, 
                        self.ligand, 
                        receptor)
            jobs.append(job)
        return jobs



@dataclass
class DINCThreadData:

    ligand: DINCMolecule
    fragment: Optional[DINCFragment]
    receptor: DINCReceptor

    replica: int = 0
    iterative_step: int = 0
    docking_type: DINC_JOB_TYPE = DINC_CORE_PARAMS.job_type



@dataclass
class DINCThreadElem:
    data: DINCThreadData
    job_info: DINCJobInfo

    @property
    def info(self):
        return pd.DataFrame({"job_name": [self.job_info.job_name],
                "receptor_name": [self.job_info.receptor_name],
                "ligand_name": [self.job_info.ligand_name],
                "replica": [self.data.replica],
                "fragment_id": [self.data.iterative_step],
                "fragment_cnt": [len(self.data.fragment.fragments) \
                                 if self.data.fragment is not None \
                                    else 1]
                            })

