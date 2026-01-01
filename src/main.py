__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"

import os
NUMPY_SINGLE_THREAD = True
if NUMPY_SINGLE_THREAD:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"   
import numpy as np
import time
from utils import pickle_objects, unpickle_objects, cpu_and_system_props, gpu_props, dict_to_str, Logger, experiment_hash_str
import jcr
from pprint import pprint
c_props = cpu_and_system_props()
g_props = gpu_props()                                             

UNPICKLE_MDP = True
            
if __name__ == "__main__":
    print("JACK'S CAR RENTAL STARTING...")
    if UNPICKLE_MDP:
        [P] = unpickle_objects(jcr.MDP_PATH)
    else:
        P = jcr.mdp_joint_distr_jcr()
        pickle_objects(jcr.MDP_PATH, [P])
    print(f"JOINT DISTRIBUTION -> SHAPE: {P.shape}, TYPE: {P.dtype}")    
    # V, policy = jcr.jcr_pi_contraction_numpy(P, plots=False)            
    V, policy, i_eval_total, time_ = jcr.jcr_pi_contraction_cuda_atomicmaxglosten(P, plots=False)
    
    # jcr.plot_value_and_policy_jcr(V, policy)
    print("JACK'S CAR RENTAL DONE.")