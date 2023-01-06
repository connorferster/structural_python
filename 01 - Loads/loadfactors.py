from typing import NamedTuple, Any
import json
import numpy as np

class Load(NamedTuple):
    """
    A class to describe loads according to the Canadian National
    Building Code (NBCC).
    
    The loads themselves can be of any type as long as they play
    well with either scalar arithmetic operators or numpy array
    operators.
    
    If array values are used, all arrays should be the same length.
    """
    D: Any = 0.0
    L: Any = 0.0
    S: Any = 0.0
    W: Any = 0.0
    E: Any = 0.0
    
## Examples

L0 = Load() # An empty load
L1 = Load(D=2.3, L=2.4, S=0.9) # Not all fields need be entered
L2 = Load( # A load with array values
    D=np.array([0.6, 1.1, 0.4]), 
    L=np.array([2.4, 3.6, 0.8]),
    S=np.array([0.5, 1.2, 3.6]),
    W=np.zeros(3),
    E=np.zeros(3),
)


def open_load_combinations(filename='NBCC_vec.json') -> dict:
    """
    Returns a dict representing the load combinations contained
    in 'filename'.
    """
    with open('NBCC_vec.json', 'r') as json_file:
        nbcc_vec = json.load(json_file)
    return nbcc_vec


def factored_max(load: Load, load_combinations: dict) -> Any:
    """
    Returns a value representing the maximum factored load of 'load' calculated
    from all of the load combinations in 'load_combinations'.
    
    If 'load' contains scalar values, the maximum factored load is a scalar maximum.
    If 'load' contains array values, the maximum factored load is the envelope of
    maximum values across the input arrays in 'load'.
    """
    fl = float('-inf') # acc
    for load_combination, load_factors in load_combinations.items():
        current_fl = factor_load(
            np.array(load).T, 
            np.array(load_factors)
        )
        fl = np.maximum(current_fl, fl)
    return fl


def factored_min(load: Load, load_combinations: dict) -> Any:
    """
    Returns a value representing the minimum factored load of 'load' calculated
    from all of the load combinations in 'load_combinations'.
    
    If 'load' contains scalar values, the minimum factored load is a scalar minimum.
    If 'load' contains array values, the minimum factored load is the envelope of
    minimum values across the input arrays in 'load'.
    """
    fl = float('inf') # acc
    for load_combination, load_factors in load_combinations.items():
        current_fl = factor_load(
            np.array(load).T, 
            np.array(load_factors)
        )
        fl = np.minimum(current_fl, fl)
    return fl


def factored_max_trace(load: Load, load_combinations: dict) -> Any:
    """
    Returns a value representing the maximum factored load of 'load' calculated
    from all of the load combinations in 'load_combinations'.
    
    If 'load' contains scalar values, the maximum factored load is a scalar maximum.
    If 'load' contains array values, the maximum factored load is the envelope of
    maximum values across the input arrays in 'load'.
    """
    factored_matrix = get_factored_matrix(load, load_combinations)
    max_trace = np.argmax(factored_matrix, axis=0)
    return max_trace


def factored_min_trace(load: Load, load_combinations: dict) -> Any:
    """
    Returns a value representing the minimum factored load of 'load' calculated
    from all of the load combinations in 'load_combinations'.
    
    If 'load' contains scalar values, the minimum factored load is a scalar minimum.
    If 'load' contains array values, the minimum factored load is the envelope of
    minimum values across the input arrays in 'load'.
    """
    factored_matrix = get_factored_matrix(load, load_combinations)
    min_trace = np.argmin(factored_matrix, axis=0)
    return min_trace


def get_factored_matrix(load: Load, load_combinations: dict) -> np.ndarray:
    """
    Returns either a 1D or 2D array representing the loads in 'load' being
    factored by 'load_combinations'. Each factored load combination is a
    row in the resulting matrix.
    
    If the values in 'load' are scalars, then a 1D matrix is returned
    If the values in 'load' are 1D arrays, then a 2D matrix is returned
    """
    acc = None
    for idx, (load_combination, load_factors) in enumerate(
        load_combinations.items()
    ):
        current_fact = factor_load(
            np.array(load).T, np.array(load_factors)
        )
        if idx == 0: acc = np.array([current_fact])
        else: acc = np.concatenate([acc, [current_fact]])
    return acc


def factor_load(load_vector: np.ndarray, factor_vector: np.ndarray) -> float:
    """
    Returns the loads in 'load_vector' factored by 'factor_vector'.
    Both arrays must be the same length.
    """
    return load_vector @ factor_vector


def alias_to_service_loads(alias_loads: dict, alias_lookup: dict) -> Load:
    """
    Returns a dictionary of service loads (i.e. loads conforming to the
    code categories of DL, LL, SL, etc.) obtained from the given loads
    in 'alias_loads' correlated against the load types in 'alias_lookup'.
    
    If the load type in 'alias_loads' is not listed in 'alias_lookup', no 
    transformation takes place.
    """
    service_loads = {}
    for load_name, load_mag in alias_loads.items():
        load_type = alias_lookup.get(load_name, load_name)
        
        if load_type in service_loads:
            service_loads[load_type] = service_loads[load_type] + load_mag
        else:
            service_loads.update({load_type: load_mag})
            
    return Load(**service_loads)


