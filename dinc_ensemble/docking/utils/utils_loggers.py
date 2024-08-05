
from dinc_ensemble import DINC_CORE_PARAMS
from pathlib import Path
import sys

import logging
logger = logging.getLogger('dinc_ensemble.docking.run')
logger.setLevel(logging.DEBUG)

def init_logging(verbosity: int, out_dir: str):

    # Set stdout logging depending on verbosity argument
    if verbosity == 1 or verbosity == 2:
        handler = logging.StreamHandler(sys.stdout)
        if verbosity == 1:
            handler.setLevel(logging.INFO)
        if verbosity == 2:
            handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Always set the logging to the log file
    fh = logging.FileHandler(str(Path(out_dir) / Path('dinc_ensemble_run.log')))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)