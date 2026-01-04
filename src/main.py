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
from numba import cuda
import time
from utils import pickle_objects, unpickle_objects, cpu_and_system_props, gpu_props, dict_to_str, Logger, experiment_hash_str
import jcr
from pprint import pprint
c_props = cpu_and_system_props()
g_props = gpu_props()                                             

# experiment settings
MAKE_JCR_MDP_JOINT_DISTR = False # False implies the distribution shall be read from a .pkl file prepared earlier
SEED = 1
GAMMA = 0.9
EPS = 1e-2
TOLERANCE_V = 1e-7
PLOTS = False
                     
def jcr_setup(make_distr):
    jcr_info = {
        "MAX_CARS_AT_LOC": jcr.MAX_CARS_AT_LOC,
        "MAX_CARS_MOVED": jcr.MAX_CARS_MOVED,
        "REWARD_CAR_RENTED": jcr.REWARD_CAR_RENTED,
        "REWARD_CAR_MOVED": jcr.REWARD_CAR_MOVED,
        "REQUEST_POISSON_LAMBDAS_AT_LOC": jcr.REQUEST_POISSON_LAMBDAS_AT_LOC,
        "RETURN_POISSON_LAMBDAS_AT_LOC": jcr.RETURN_POISSON_LAMBDAS_AT_LOC,
        }
    print(f"JCR INFO:\n{dict_to_str(jcr_info)}")
    print(f"JCR MDP JOINT DISTRIBUTION...")
    P = None
    if make_distr:
        P = jcr.make_jcr_mdp_joint_distr()
        pickle_objects(jcr.MDP_PATH, [P])
    else:
        [P] = unpickle_objects(jcr.MDP_PATH)        
    t1_copy_p = time.time()
    dev_P = cuda.to_device(P)
    cuda.synchronize() 
    t2_copy_p = time.time()
    print(f"JCR MDP JOINT DISTRIBUTION READY. [shape: {P.shape}, type: {P.dtype}, copied to device (gpu) in: {t2_copy_p - t1_copy_p} s]")    
    return jcr_info, P, dev_P
            
if __name__ == "__main__":
    
    experiment_info = {} # TODO
    
    print("JACK'S CAR RENTAL STARTING...")
    t1 = time.time()
    line_separator = 244 * "="
    
    jcr_info, P, dev_P = jcr_setup(MAKE_JCR_MDP_JOINT_DISTR)
    print(line_separator)         
      
    print(f"HASH STRING: TODO")    
    print(line_separator)          
    print(f"EXPERIMENT INFO:\n{dict_to_str(experiment_info)}")
    print(line_separator)    
    print("CPU AND SYSTEM:")
    pprint(c_props)
    print("GPU:")
    pprint(g_props)
    print(line_separator)        
    
    np.random.seed(SEED)
    V_in = np.zeros(jcr.STATES.shape[0], dtype=np.float32)
    policy_in = np.random.randint(-jcr.MAX_CARS_MOVED, jcr.MAX_CARS_MOVED + 1, size=jcr.STATES.shape[0], dtype=np.int16) + jcr.MAX_CARS_MOVED
    print(f"RANDOM INITIAL POLICY [shape: {policy_in.shape}]:\n{jcr.ACTIONS[policy_in]}")
    print(f"V FOR INITIAL POLICY ZEROED.")
    print(line_separator)    
    
    policy_out, V_out, d, k_main, k_eval_total, time_ = jcr.jcr_pi_contraction_numpy(
         policy_in, V_in, P,
         gamma=GAMMA, eps=EPS, tolerance_v=TOLERANCE_V,      
         states=jcr.STATES, actions=jcr.ACTIONS, rewards_rental=jcr.REWARDS_RENTAL, reward_car_moved=jcr.REWARD_CAR_MOVED,    
         verbose=True, verbose_iters=False, plots=PLOTS)     

    policy_out, V_out, d, k_main, k_eval_total, time_ = jcr.jcr_pi_contraction_cuda_atomicmaxglosten(
         policy_in, V_in, dev_P,
         gamma=GAMMA, eps=EPS, tolerance_v=TOLERANCE_V,
         states=jcr.STATES, actions=jcr.ACTIONS, rewards_rental=jcr.REWARDS_RENTAL, reward_car_moved=jcr.REWARD_CAR_MOVED,                 
         lazy_stop_check=jcr.DEFAULT_LAZY_STOP_CHECK, tpb=jcr.DEFAULT_TPB, verbose=True, verbose_iters=False, plots=PLOTS)
    
    policy_out, V_out, d, k_main, k_eval_total, time_ = jcr.jcr_pi_contraction_cuda_atomicmax(
         policy_in, V_in, dev_P,
         gamma=GAMMA, eps=EPS, tolerance_v=TOLERANCE_V,
         states=jcr.STATES, actions=jcr.ACTIONS, rewards_rental=jcr.REWARDS_RENTAL, reward_car_moved=jcr.REWARD_CAR_MOVED,                 
         lazy_stop_check=jcr.DEFAULT_LAZY_STOP_CHECK, tpb=jcr.DEFAULT_TPB, verbose=True, verbose_iters=False, plots=PLOTS)    
    
    policy_out, V_out, d, k_main, k_eval_total, time_ = jcr.jcr_pi_contraction_cuda_reducemax(
        policy_in, V_in, dev_P,
        gamma=GAMMA, eps=EPS, tolerance_v=TOLERANCE_V,
        states=jcr.STATES, actions=jcr.ACTIONS, rewards_rental=jcr.REWARDS_RENTAL, reward_car_moved=jcr.REWARD_CAR_MOVED,                 
        lazy_stop_check=jcr.DEFAULT_LAZY_STOP_CHECK, tpb=jcr.DEFAULT_TPB, verbose=True, verbose_iters=False, plots=PLOTS)    
    
    policy_out, V_out, d, k_main, k_eval_total, time_ = jcr.jcr_pi_contraction_cuda_gridsync(
        policy_in, V_in, dev_P,
        gamma=GAMMA, eps=EPS, tolerance_v=TOLERANCE_V,
        states=jcr.STATES, actions=jcr.ACTIONS, rewards_rental=jcr.REWARDS_RENTAL, reward_car_moved=jcr.REWARD_CAR_MOVED,                 
        tpb=jcr.DEFAULT_TPB, verbose=True, verbose_iters=False, plots=PLOTS)
    
    if PLOTS:   
        jcr.plot_value_and_policy_jcr(V_out, policy_out)        
    t2 = time.time()    
    print(f"JACK'S CAR RENTAL DONE. [hash string: TODO, time: {t2 - t1} s]")