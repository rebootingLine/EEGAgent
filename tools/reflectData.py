from tools import function_register
from .polar import name2index
from typing import List
from .registerData import getRegisteredData

@function_register.register(
    description="This function should only be called when no effective information can be obtained from other tools. Directly extract a 1-second segment of raw data.",
    parameters=[
        {
            "name": "name",
            "type": "List[str]",
            "description": "List of channel names to extract, e.g., ['FP1-F7', 'F7-T3']"
        },
        {
            "name": "start",
            "type": "int",
            "description": "Start time in seconds"
        },
        {
            "name": "end",
            "type": "int",
            "description": "End time in seconds"
        }
    ],
    returns={
        "type": "Dict[str, np.ndarray]",
        "description": "A dictionary with channel names as keys and corresponding EEG data arrays as values"
    }
)
def reflectData(name: List[str], start: int, end: int, config):
    if end - start > 1:
        raise ValueError("The time interval between start and end should not exceed 10 seconds.")
    data = getRegisteredData()
    fs = config['fs']
    picks = [name2index[ch_name] for ch_name in name] 
    data = data[picks, start*fs : end*fs]
    result = {
        name[i]: data[i] for i in range(len(name))
    }
    
    return result

    