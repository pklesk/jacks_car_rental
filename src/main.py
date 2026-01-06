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
import sys       
import numpy as np
from numba import cuda
import time
from utils import pickle_objects, unpickle_objects, cpu_and_system_props, gpu_props, dict_to_str, Logger, experiment_hash_str
import jcr
from pprint import pprint
c_props = cpu_and_system_props()
g_props = gpu_props()
                                           
# global settings                
FOLDER_EXPERIMENTS = "../experiments/"
DEFAULT_REPETITIONS = 10

# experiment settings
MAKE_JCR_MDP_JOINT_DISTR = False # False implies the distribution shall be read from a .pkl file prepared earlier
SEED = 1
GAMMA = 0.9
EPS = 1e-4
TOLERANCE_V = 1e-7
PLOTS = False

APPROACHES_PI_CONTRACTION = { # approaches for contraction iteration (policy iteration)
    jcr.jcr_pi_contraction_cpu_numpy.__name__: (True, jcr.jcr_pi_contraction_cpu_numpy, DEFAULT_REPETITIONS, {}),
    jcr.jcr_pi_contraction_cuda_atomicmax.__name__: (True, jcr.jcr_pi_contraction_cuda_atomicmax, DEFAULT_REPETITIONS, {"lazy_stop_check": jcr.DEFAULT_LAZY_STOP_CHECK, "tpb": jcr.DEFAULT_TPB}),
    jcr.jcr_pi_contraction_cuda_atomicmaxglosten.__name__: (True, jcr.jcr_pi_contraction_cuda_atomicmaxglosten, DEFAULT_REPETITIONS, {"lazy_stop_check": jcr.DEFAULT_LAZY_STOP_CHECK, "tpb": jcr.DEFAULT_TPB}),    
    jcr.jcr_pi_contraction_cuda_reducemax.__name__: (True, jcr.jcr_pi_contraction_cuda_reducemax, DEFAULT_REPETITIONS, {"lazy_stop_check": jcr.DEFAULT_LAZY_STOP_CHECK, "tpb": jcr.DEFAULT_TPB}),
    jcr.jcr_pi_contraction_cuda_gridsync.__name__: (True, jcr.jcr_pi_contraction_cuda_gridsync, DEFAULT_REPETITIONS, {"tpb": jcr.DEFAULT_TPB})
    }
                               
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
        if os.path.exists(jcr.MDP_PATH):
            [P] = unpickle_objects(jcr.MDP_PATH)
    if P is None:
        print(f"FILE {jcr.MDP_PATH} WITH JCR MDP JOINT DISTRIBUTION NOT PRESENT.")
        print(f"CHANGE GLOBAL SETTING 'MAKE_JCR_MDP_JOINT_DISTR' TO TRUE AND RUN AGAIN.")
        return jcr_info, None, None        
    t1_copy_p = time.time()
    dev_P = cuda.to_device(P)
    cuda.synchronize() 
    t2_copy_p = time.time()
    print(f"JCR MDP JOINT DISTRIBUTION READY. [shape: {P.shape}, entries: {P.size}, type: {P.dtype}, {P.nbytes/1024**3:.2f} GiB copied to device (gpu) in: {t2_copy_p - t1_copy_p} s]")    
    return jcr_info, P, dev_P

def approaches_info(approaches):
    info = {}
    for key in approaches.keys():
        if approaches[key][0]:
            info[key]  = (approaches[key][0], approaches[key][1].__name__, approaches[key][2], approaches[key][3])
        else:
            info[key] = (approaches[key][0], approaches[key][1].__name__, 0, {})
    return info

def gpu_warmup(policy, V, dev_P):
    jcr.jcr_pi_contraction_cuda_atomicmax(policy, V, dev_P, gamma=0.1, eps=1e2, tolerance_v=1e2, verbose=False)
    cuda.synchronize()
    print("GPU WARMED UP.")
            
# --------------------------------------------------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------------------------------------------------            
if __name__ == "__main__":
    
    experiment_info = {
        "SEED": SEED,
        "JCR_MAX_CARS_AT_LOC": jcr.MAX_CARS_AT_LOC,
        "JCR_MAX_CARS_MOVED": jcr.MAX_CARS_MOVED,
        "JCR_REWARD_CAR_RENTED": jcr.REWARD_CAR_RENTED,
        "JCR_REWARD_CAR_MOVED": jcr.REWARD_CAR_MOVED,
        "JCR_REQUEST_POISSON_LAMBDAS_AT_LOC": jcr.REQUEST_POISSON_LAMBDAS_AT_LOC,
        "JCR_RETURN_POISSON_LAMBDAS_AT_LOC": jcr.RETURN_POISSON_LAMBDAS_AT_LOC,
        "NUMPY_SINGLE_THREAD": NUMPY_SINGLE_THREAD,        
        "GAMMA": GAMMA,
        "EPS": EPS,
        "TOLERANCE_V": TOLERANCE_V,
        "PLOTS": PLOTS,
        **approaches_info(APPROACHES_PI_CONTRACTION)
        }
    experiment_hs = experiment_hash_str(experiment_info, c_props, g_props)
    
    logger = Logger(f"{FOLDER_EXPERIMENTS}{experiment_hs}.log")    
    sys.stdout = logger
    
    print("JACK'S CAR RENTAL STARTING...")
    t1 = time.time()
    line_separator = 252 * "="         
    
    print(f"HASH STRING: {experiment_hs}")    
    print(line_separator)          
    print(f"EXPERIMENT INFO:\n{dict_to_str(experiment_info)}")
    print(line_separator)    
    print("CPU AND SYSTEM:")
    pprint(c_props)
    print("GPU:")
    pprint(g_props)
    print(line_separator)        
    
    jcr_info, P, dev_P = jcr_setup(MAKE_JCR_MDP_JOINT_DISTR)
    if P is None:
        sys.exit()
    print(line_separator)
    
    np.random.seed(SEED)
    V_in = np.zeros(jcr.STATES.shape[0], dtype=np.float32)
    print(f"V FOR INITIAL POLICY ZEROED. [shape: {V_in.shape}]")
    policy_in = np.random.randint(-jcr.MAX_CARS_MOVED, jcr.MAX_CARS_MOVED + 1, size=jcr.STATES.shape[0], dtype=np.int16) + jcr.MAX_CARS_MOVED
    print(f"RANDOM INITIAL POLICY [shape: {policy_in.shape}]:\n{jcr.ACTIONS[policy_in]}")    
    if PLOTS:
        jcr.plot_value_and_policy_jcr(V_in, policy_in)        

    # about to execute policy iteration contraction approaches
    pi_contraction_ref_approach_name = None
    pi_contraction_ref_V_out = None
    pi_contraction_ref_policy_out = None    
    pi_contraction_ref_time_mean = None
    pi_contraction_times = {}
    pi_contraction_ds = {}
    pi_contraction_ks_main = {}
    pi_contraction_ks_eval_total = {}        
    gpu_warmed_up = False
    for index, (approach_name, (approach_on, approach_function, approach_repetitions, approach_extra_params)) in enumerate(APPROACHES_PI_CONTRACTION.items()):
        if approach_on:
            print(line_separator)
            print(f"PI CONTRACTION APPROACH {index + 1}: {approach_name}...", flush=True)
            P_ = P if approach_name == jcr.jcr_pi_contraction_cpu_numpy.__name__ else dev_P
            if "cuda" in approach_name and not gpu_warmed_up:
                gpu_warmup(policy_in, V_in, P_)
                gpu_warmed_up = True
            for r in range(approach_repetitions):
                print("---")               
                print(f"REPETITION: {r + 1}/{approach_repetitions}:")
                policy_out, V_out, d, k_main, k_eval_total, time_, hfp = approach_function(          
                    policy_in, V_in, P_,
                    gamma=GAMMA, eps=EPS, tolerance_v=TOLERANCE_V,      
                    states=jcr.STATES, actions=jcr.ACTIONS, rewards_rental=jcr.REWARDS_RENTAL, reward_car_moved=jcr.REWARD_CAR_MOVED,
                    **approach_extra_params,
                    plots=PLOTS)  
                if approach_name not in pi_contraction_times:
                    pi_contraction_times[approach_name] = []
                    pi_contraction_ds[approach_name] = []
                    pi_contraction_ks_main[approach_name] = []
                    pi_contraction_ks_eval_total[approach_name] = []
                pi_contraction_times[approach_name].append(time_)
                pi_contraction_ds[approach_name].append(d)                
                pi_contraction_ks_main[approach_name].append(k_main)
                pi_contraction_ks_eval_total[approach_name].append(k_eval_total)             
            time_mean = np.mean(pi_contraction_times[approach_name])
            time_std = np.std(pi_contraction_times[approach_name])
            d_mean = np.mean(pi_contraction_ds[approach_name])
            k_main_mean = np.mean(pi_contraction_ks_main[approach_name])
            k_eval_total_mean = np.mean(pi_contraction_ks_eval_total[approach_name])
            if pi_contraction_ref_approach_name is None:
                pi_contraction_ref_approach_name = approach_name
                pi_contraction_ref_V_out = V_out
                pi_contraction_ref_policy_out = policy_out
                pi_contraction_ref_time_mean = time_mean                            
            d_vs_ref = np.max(np.abs(V_out - pi_contraction_ref_V_out))
            d_policy_vs_ref = np.max(np.abs(jcr.ACTIONS[policy_out] - jcr.ACTIONS[pi_contraction_ref_policy_out]))
            speedup_vs_ref = pi_contraction_ref_time_mean / time_mean
            print("***")
            print("SUMMARY:")
            print(f"V OUT:\n{V_out}")
            print(f"POLICY OUT:\n{jcr.ACTIONS[policy_out]}")            
            print(f"D_INF OF V VS REF: {str(d_vs_ref)}")
            print(f"D_INF OF POLICY VS REF: {str(d_policy_vs_ref)}")            
            print(f"D_INF (AT STOP) MEAN: {d_mean}")
            print(f"MAIN ITERATIONS MEAN: {k_main_mean}")
            print(f"EVAL ITERATIONS TOTAL MEAN: {k_eval_total_mean}")                    
            print(f"TIME MEAN: {time_mean} s, STD: {time_std} s, STD_%: {(time_std / time_mean) * 100:.1f}%")                        
            print(f"SPEEDUP VS REF: {speedup_vs_ref}")            
            if PLOTS: 
                method_name = approach_function.__name__
                overal_title = f"COMPUTED BY: {method_name}"
                overal_title += f"\n[$d_{{\\infty}}$: {d:.3e}, main iters.: {k_main}, evaluation total iters.: {k_eval_total}]"
                jcr.plot_value_and_policy_jcr(V_out, policy_out, overall_title=overal_title)
                                                                                  
    print(line_separator)
    print(line_separator)
    print("FINAL SUMMARY:")
    for index, (approach_name, (approach_on, approach_function, approach_repetitions, approach_extra_params)) in enumerate(APPROACHES_PI_CONTRACTION.items()):
        if approach_on:
            if approach_name not in pi_contraction_times:
                print(f"PI CONTRACTION APPROACH {index + 1}: {approach_name} SKIPPED.")
                continue
            reference_info = " (REFERENCE)" if approach_name == pi_contraction_ref_approach_name else ""
            k_main_mean = np.mean(pi_contraction_ks_main[approach_name])
            k_eval_total_mean = np.mean(pi_contraction_ks_eval_total[approach_name])
            d_mean = np.mean(pi_contraction_ds[approach_name])
            time_mean = np.mean(pi_contraction_times[approach_name])
            time_std = np.std(pi_contraction_times[approach_name])
            speedup = pi_contraction_ref_time_mean / time_mean 
            print(f"PI CONTRACTION APPROACH {index + 1}: {approach_name}{reference_info} -> MEAN MAIN ITERS: {k_main_mean}, MEAN EVAL ITERS TOTAL: {k_eval_total_mean}, MEAN D_INF: {d_mean}, MEAN TIME: {time_mean} s, TIME STD: {time_std} s, STD_%: {(time_std / time_mean) * 100:.1f}%, SPEED-UP: {speedup:.2f}", flush=True)
        else:
            print(f"PI CONTRACTION APPROACH {index + 1}: {approach_name} OFF.")        
    t2 = time.time()    
    print(f"JACK'S CAR RENTAL DONE. [hash string: {experiment_hs}, time: {t2 - t1} s]")
    sys.stdout = sys.__stdout__
    logger.logfile.close()                                          
