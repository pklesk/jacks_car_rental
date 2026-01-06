__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"

import cpuinfo
import platform
import psutil
from numba import cuda
import sys
import pickle
import time
 
def pickle_objects(fname, some_list):
    print(f"PICKLE OBJECTS... [to file: {fname}]")
    t1 = time.time()
    try:
        f = open(fname, "wb+")
        pickle.dump(some_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
    except IOError:
        sys.exit("[error occurred when trying to open or pickle the file]")
    t2 = time.time()
    print(f"PICKLE OBJECTS DONE. [time: {t2 - t1} s]")

def unpickle_objects(fname):
    print(f"UNPICKLE OBJECTS... [from file: {fname}]")
    t1 = time.time()
    try:    
        f = open(fname, "rb")
        some_list = pickle.load(f)
        f.close()
    except IOError:
        sys.exit("[error occurred when trying to open or read the file]")
    t2 = time.time()
    print(f"UNPICKLE OBJECTS DONE. [time: {t2 - t1} s]")
    return some_list
  
def dict_to_str(d, indent=0):
    """Returns a vertically formatted string representation of a dictionary."""
    indent_str = indent * " "
    dict_str = indent_str + "{"
    for i, key in enumerate(d):
        dict_str += "\n" + indent_str + "  "  + str(key) + ": " + str(d[key]) + ("," if i < len(d) - 1 else "")    
    dict_str += "\n" + indent_str + "}"
    return dict_str

def cpu_and_system_props():
    """Returns a dictionary with properties of CPU and OS."""
    props = {}    
    info = cpuinfo.get_cpu_info()
    un = platform.uname()
    props["cpu_name"] = info["brand_raw"]
    props["ram_size"] = f"{psutil.virtual_memory().total / 1024**3:.1f} GB"
    props["os_name"] = f"{un.system} {un.release}"
    props["os_version"] = f"{un.version}"
    props["os_machine"] = f"{un.machine}"    
    return props    

def gpu_props():
    """Returns a dictionary with properties of GPU device."""
    gpu = cuda.get_current_device()
    props = {}
    props["name"] = gpu.name if isinstance(gpu.name, str) else gpu.name.decode("ASCII")
    props["max_threads_per_block"] = gpu.MAX_THREADS_PER_BLOCK
    props["max_block_dim_x"] = gpu.MAX_BLOCK_DIM_X
    props["max_block_dim_y"] = gpu.MAX_BLOCK_DIM_Y
    props["max_block_dim_z"] = gpu.MAX_BLOCK_DIM_Z
    props["max_grid_dim_x"] = gpu.MAX_GRID_DIM_X
    props["max_grid_dim_y"] = gpu.MAX_GRID_DIM_Y
    props["max_grid_dim_z"] = gpu.MAX_GRID_DIM_Z    
    props["max_shared_memory_per_block"] = gpu.MAX_SHARED_MEMORY_PER_BLOCK
    props["async_engine_count"] = gpu.ASYNC_ENGINE_COUNT
    props["can_map_host_memory"] = gpu.CAN_MAP_HOST_MEMORY
    props["multiprocessor_count"] = gpu.MULTIPROCESSOR_COUNT
    props["warp_size"] = gpu.WARP_SIZE
    props["unified_addressing"] = gpu.UNIFIED_ADDRESSING
    props["pci_bus_id"] = gpu.PCI_BUS_ID
    props["pci_device_id"] = gpu.PCI_DEVICE_ID
    props["compute_capability"] = gpu.compute_capability            
    CC_CORES_PER_SM_DICT = {
        (2, 0): 32,
        (2, 1): 48,
        (3, 0): 256,
        (3, 5): 256,
        (3, 7): 256,
        (5, 0): 128,
        (5, 2): 128,
        (6, 0): 64,
        (6, 1): 128,
        (7, 0): 64,
        (7, 5): 64,
        (8, 0): 64,
        (8, 6): 128,
        (8, 7): 128,
        (8, 9): 128,
        (9, 0): 128,
        (12, 0) : 128
        }
    props["cores_per_SM"] = CC_CORES_PER_SM_DICT.get(gpu.compute_capability)
    props["cores_total"] = props["cores_per_SM"] * gpu.MULTIPROCESSOR_COUNT
    return props

def hash_function(s):
    """Returns a hash code (integer) for given string as a base 31 expansion."""
    h = 0
    for c in s:
        h *= 31 
        h += ord(c)
    return h

def hash_str(params, digits):
    return str((hash_function(str(params)) & ((1 << 32) - 1)) % 10**digits).rjust(digits, "0") 

def experiment_hash_str(experiment_info, c_props, g_props, all_hs_digits=10, experiment_hs_digits=5, env_hs_digits=3):
    """Returns a hash string for an experiment, based on its settings and properties."""
    experiment_hs =  hash_str(experiment_info, digits=experiment_hs_digits)    
    env_props = {**c_props, **g_props}    
    env_hs =  hash_str(env_props, digits=env_hs_digits)
    all_info = {**experiment_info, **env_props}
    all_hs = hash_str(all_info, digits=all_hs_digits)
    approaches_flags_str = ""
    suffix = f"{experiment_info['SEED']};{experiment_info['JCR_MAX_CARS_AT_LOC']};{experiment_info['JCR_MAX_CARS_MOVED']};{experiment_info['JCR_REWARD_CAR_RENTED']};{experiment_info['JCR_REWARD_CAR_MOVED']};"
    suffix += f"{experiment_info['JCR_REQUEST_POISSON_LAMBDAS_AT_LOC']};{experiment_info['JCR_RETURN_POISSON_LAMBDAS_AT_LOC']};"    
    suffix += f"{experiment_info['NUMPY_SINGLE_THREAD']};{experiment_info['GAMMA']};{experiment_info['EPS']:.1e};{experiment_info['TOLERANCE_V']:.1e};{experiment_info['PLOTS']};"
    suffix = suffix.replace(" ", "")
    for key in experiment_info.keys():
        if key.startswith("jcr_pi_contraction_"):
            approaches_flags_str += "T" if experiment_info[key][0] else "F"
    suffix += f"{approaches_flags_str}"
    hs = f"{all_hs}_{experiment_hs}_{env_hs}_[{suffix}]"
    return hs

class Logger:
    """Class for simultaneous logging to console and a log file (for purposes of experiments)."""
    def __init__(self, fname):
        """Constructor of ``Logger`` instances."""
        self.logfile = open(fname, "w", encoding="utf-8")  
        
    def write(self, message):
        """Writes a message to console and a log file.""" 
        self.logfile.write(message)
        self.logfile.flush() 
        sys.__stdout__.write(message)

    def flush(self):
        """Empty function required for buffering."""
        pass  