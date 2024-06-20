from enum import Enum
from dinc_ensemble.parameters import VINA_ENGINE_PARAMS, SCORE_F

class LigandCheckType(Enum):
    ESSENTIAL = 0
    AD4 = 1
    
    def check_condition(self) -> bool:
        if self == LigandCheckType.ESSENTIAL:
            return True
        if self == LigandCheckType.AD4:
            return (VINA_ENGINE_PARAMS.score_f == SCORE_F.AD4)
        return False