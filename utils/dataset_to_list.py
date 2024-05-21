import sys
import os

from typing import List

# sys.path.append('.')
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'resources'))
from rnn_training import DatasetRNN
from bandits import BanditSession

def dataset_to_list(dataset: DatasetRNN) -> List[BanditSession] :
    """Transforms a dataset (shape -> (sessions, trials, variables)) into a list of experiments [BanditSession]

    Args:
        dataset (DatasetRNN): Dataset containing the variables
    
    Return:
        List[BanditSession]: ...
    
    """  
    
    raise NotImplementedError("Conversion is not possible right now because DatasetRNN does not carry information for BanditSession.timeseries and BanditSession.q")
    