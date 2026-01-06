__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"

import numpy as np
from numpy import inf
import math
import itertools
import time
from numba import cuda
from numba import void, int8, int16, int32, float32, boolean
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
import os
os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"] = "1"
from matplotlib import pyplot as plt
from scipy.ndimage import label

# global settings
MAX_CARS_AT_LOC = 20
MAX_CARS_MOVED = 5
REWARD_CAR_RENTED = np.float32(10.0)
REWARD_CAR_MOVED = np.float32(-2.0)
REQUEST_POISSON_LAMBDAS_AT_LOC = [3.0, 4.0]
RETURN_POISSON_LAMBDAS_AT_LOC = [3.0, 2.0]

# constants implied by global settings (do not manipulate)
STATES = np.array(list(itertools.product(np.arange(0, MAX_CARS_AT_LOC + 1), np.arange(0, MAX_CARS_AT_LOC + 1))), dtype=np.int16)
ACTIONS = np.arange(-MAX_CARS_MOVED, MAX_CARS_MOVED + 1, dtype=np.int16)
MAX_N = MAX_CARS_AT_LOC
REQUEST_DISTRS_AT_LOCS = [np.array([l**n * np.exp(-l) / math.factorial(n) for n in range(MAX_N + 1)]) for l in REQUEST_POISSON_LAMBDAS_AT_LOC]
RETURN_DISTRS_AT_LOCS = [np.array([l**n * np.exp(-l) / math.factorial(n) for n in range(MAX_N + 1)]) for l in RETURN_POISSON_LAMBDAS_AT_LOC]
for distrs in [REQUEST_DISTRS_AT_LOCS, RETURN_DISTRS_AT_LOCS]: 
    for i in range(len(distrs)):
        distrs[i][-1] = 1.0 - np.sum(distrs[i][:-1]) # summation of distribution to 1 
REWARDS_RENTAL = REWARD_CAR_RENTED * np.arange(2 * MAX_CARS_AT_LOC + 1, dtype=np.float32) 
MDP_PATH = f"../mdp_joint_distrs/mdp_joint_distr_jcr_{MAX_CARS_AT_LOC}_{MAX_CARS_MOVED}.pkl"

# CUDA defaults
DEFAULT_MAX_TPB = cuda.get_current_device().MAX_THREADS_PER_BLOCK // 2 
DEFAULT_TPB = 64 
DEFAULT_LAZY_STOP_CHECK = 5

# constraints
MAX_N_STATES = 1024
assert (MAX_CARS_AT_LOC + 1)**2 <= MAX_N_STATES, f"Maximum number of states {MAX_N_STATES} exceeded due to too large maximum of cars at location set to: {MAX_CARS_AT_LOC}."

def make_jcr_mdp_joint_distr(states=STATES, actions=ACTIONS, rewards_rental=REWARDS_RENTAL, request_distrs_at_locs=REQUEST_DISTRS_AT_LOCS, return_distrs_at_locs=RETURN_DISTRS_AT_LOCS, return_as_float32_array=True):
    print("MAKE JCR MDP JOINT DISTR...")
    t1 = time.time()
    max_cars_at_loc = int(np.sqrt(states.shape[0])) - 1
    P = np.zeros((states.shape[0], actions.shape[0], rewards_rental.shape[0], states.shape[0])) # float64 when being prepared, before the return can be converted to float32
    print(f"[shape of P to prepare: {P.shape}, entries: {P.size}]")
    for s_index in range(states.shape[0]):
        print(f"[progress: {s_index + 1}/{states.shape[0]}]")
        s_cars_at_0, s_cars_at_1 = states[s_index]
        for a_index in range(actions.shape[0]):
            a = actions[a_index]
            cars_moved = min(s_cars_at_0, a) if a > 0 else min(s_cars_at_1, -a)
            cars_at_0 = s_cars_at_0 - np.sign(a) * cars_moved
            cars_at_1 = s_cars_at_1 + np.sign(a) * cars_moved                
            cars_at_0 = min(cars_at_0, max_cars_at_loc) 
            cars_at_1 = min(cars_at_1, max_cars_at_loc)                                   
            for request_index in range(states.shape[0]):
                request = states[request_index]
                P_request = request_distrs_at_locs[0][request[0]] * request_distrs_at_locs[1][request[1]] 
                rented_at_0 = min(request[0], cars_at_0)
                rented_at_1 = min(request[1], cars_at_1)
                reward_index = rented_at_0 + rented_at_1                                       
                for return_index in range(states.shape[0]):
                    returned_at_0, returned_at_1 = states[return_index]
                    P_return = return_distrs_at_locs[0][returned_at_0] * return_distrs_at_locs[1][returned_at_1]                    
                    cars_at_0_next = min(cars_at_0 - rented_at_0 + returned_at_0, max_cars_at_loc)
                    cars_at_1_next = min(cars_at_1 - rented_at_1 + returned_at_1, max_cars_at_loc)
                    s_next_index = cars_at_0_next * (max_cars_at_loc + 1) + cars_at_1_next                                                              
                    P[s_index, a_index, reward_index, s_next_index] += P_request * P_return
    if return_as_float32_array:
        P = P.astype(np.float32)                                                             
    t2 = time.time()
    print(f"MAKE JCR MDP JOINT DISTR DONE. [time: {t2 - t1} s]")
    return P    

def jcr_pi_contraction_cpu_numpy(policy_in, V_in, P, 
                             gamma, eps, tolerance_v,                               
                             states=STATES, actions=ACTIONS, rewards_rental=REWARDS_RENTAL, reward_car_moved=REWARD_CAR_MOVED,
                             verbose=True, verbose_iters=False, plots=False):
    if verbose:
        print(f"JCR PI CONTRACTION CPU NUMPY... [gamma: {gamma}, eps: {eps}, tolerance_v: {tolerance_v}]")
    t1 = time.time()
    history_for_plots = []
    if verbose_iters and plots:
        history_for_plots.append((V_in, policy_in))
        plot_value_and_policy_jcr(V_in, policy_in)    
    policy = np.copy(policy_in)
    V_src = np.copy(V_in)
    V_dst = np.copy(V_in)
    k_main = 0
    k_eval_total = 0
    while True:
        if verbose_iters:
            print(f"---")        
            print(f"[main iteration {k_main + 1}:]")        
        k_eval = 0
        while True:
            t1_eval = time.time()    
            d = 0                     
            for s_index in range(states.shape[0]):
                a_index = policy[s_index]
                reward_cars_moved = np.abs(actions[a_index]) * reward_car_moved
                v_new = 0.0
                for r_index in range(rewards_rental.shape[0]):
                    r = reward_cars_moved + rewards_rental[r_index]
                    v_new += P[s_index, a_index, r_index].dot(r + gamma * V_src)             
                    # alternative code below when fma (fused mul-add) needs to be avoided for test purposes
                    # term_add = r + gamma * V_src
                    # term_mul_add = P[s_index, a_index, r_index] * term_add
                    # v_new += np.sum(term_mul_add, dtype=np.float32)
                V_dst[s_index] = v_new
            d = max(np.abs(V_src - V_dst))
            k_eval += 1
            tmp = V_src # ping-pong trick
            V_src = V_dst
            V_dst = tmp                 
            t2_eval = time.time()
            if verbose_iters:
                print(f"[policy evaluation iteration {k_eval} [d_inf: {str(d)}, time: {t2_eval - t1_eval} s]]")
            if d <= eps:
                k_eval_total += k_eval
                break
        t1_impr = time.time()
        policy_stable = True
        for s_index in range(states.shape[0]):
            a_so_far = policy[s_index]
            qs = np.zeros(actions.shape[0])
            for a_index in range(actions.shape[0]):
                reward_cars_moved = np.abs(actions[a_index]) * reward_car_moved
                q = 0.0
                for r_index in range(rewards_rental.shape[0]):
                    r = reward_cars_moved + rewards_rental[r_index]
                    q += (P[s_index, a_index, r_index].dot(r + gamma * V_src))
                qs[a_index] = q
            policy[s_index] = np.argmax(qs)
            if policy[s_index] != a_so_far:
                policy_stable = False 
        t2_impr = time.time()
        k_main += 1
        if verbose_iters:
            print(f"[policy improvement [policy_stable: {policy_stable}, d_inf: {str(d)}, d_inf <= tolerance_v: {d <= tolerance_v}, time: {t2_impr - t1_impr} s]]")        
            if plots:
                V_now = np.copy(V_src)
                policy_now = np.copy(policy)
                history_for_plots.append((V_now, policy_now))
                plot_value_and_policy_jcr(V_now, policy_now)
        if policy_stable or (k_eval == 1 and d <= tolerance_v):
            break
    V_out = V_src
    policy_out = policy
    t2 = time.time()
    if verbose:
        print(f"JCR PI CONTRACTION CPU NUMPY DONE. [d_inf: {str(d)}, main iterations: {k_main}, evaluation iterations total: {k_eval_total}, time: {t2 - t1} s]")
    return policy_out, V_out, d, k_main, k_eval_total, t2 - t1, history_for_plots                                     
    
def jcr_pi_contraction_cuda_atomicmax(policy_in, V_in, dev_P,                                             
                                      gamma, eps, tolerance_v,
                                      states=STATES, actions=ACTIONS, rewards_rental=REWARDS_RENTAL, reward_car_moved=REWARD_CAR_MOVED,                                              
                                      lazy_stop_check=DEFAULT_LAZY_STOP_CHECK, tpb=DEFAULT_TPB,
                                      verbose=True, verbose_iters=False, plots=False):
    if verbose:
        print(f"JCR PI CONTRACTION CUDA ATOMICMAX... [gamma: {gamma}, eps: {eps}, tolerance_v: {tolerance_v}, lazy_stop_check: {lazy_stop_check}, tpb: {tpb}]")
    t1 = time.time()    
    history_for_plots = []
    if verbose_iters and plots:
        history_for_plots.append((V_in, policy_in))
        plot_value_and_policy_jcr(V_in, policy_in)                
    dev_V_in = cuda.to_device(V_in)
    dev_V_out = cuda.to_device(V_in)
    dev_policy = cuda.to_device(policy_in)
    d = np.zeros(1, dtype=np.float32)
    dev_d = cuda.to_device(d)
    dev_policy_stable = cuda.to_device(np.ones(1, dtype=np.int32))            
    bpg = states.shape[0]
    if verbose:
        print(f"[bpg: {bpg} (implied by number of states)]")
    k_main = 0
    k_eval_total = 0        
    while True:
        if verbose_iters:
            print(f"---")
            print(f"[main iteration {k_main + 1}:]")        
        k_eval = 0
        while True:
            t1_eval = time.time()
            jcr_pi_contraction_cuda_atomicmax_dreset[1, 1](dev_d)                        
            jcr_pi_contraction_cuda_atomicmax_eval[bpg, tpb](dev_P, dev_V_in, dev_policy, reward_car_moved, gamma, dev_V_out, dev_d)            
            t2_eval = time.time()
            k_eval += 1
            tmp = dev_V_in # ping-pong trick
            dev_V_in = dev_V_out
            dev_V_out = tmp
            if k_eval % lazy_stop_check == 0:        
                dev_d.copy_to_host(ary=d)
                cuda.synchronize()
                if verbose_iters: 
                    print(f"[policy evaluation iteration {k_eval} [d_inf: {str(d[0])}, time: {t2_eval - t1_eval} s]]")            
                if d[0] <= eps:
                    k_eval_total += k_eval
                    break
        t1_impr = time.time()
        jcr_pi_contraction_cuda_atomicmax_psreset[1, 1](dev_policy_stable)                   
        jcr_pi_contraction_cuda_atomicmax_improve[bpg, tpb](dev_P, dev_V_in, reward_car_moved, gamma, dev_policy, dev_policy_stable)            
        policy_stable = dev_policy_stable.copy_to_host()[0]
        policy_stable = True if policy_stable == 1 else False
        cuda.synchronize()
        t2_impr = time.time()
        k_main += 1
        if verbose_iters:
            print(f"[policy improvement [policy_stable: {policy_stable}, d_inf: {str(d[0])}, d_inf <= tolerance_v: {d[0] <= tolerance_v}, time: {t2_impr - t1_impr} s]]")        
            if plots:                
                V_now = dev_V_in.copy_to_host()
                policy_now = dev_policy.copy_to_host()
                cuda.synchronize()
                history_for_plots.append((V_now, policy_now))                
                plot_value_and_policy_jcr(V_now, policy_now)
        if policy_stable or (k_eval == 1 and d[0] <= tolerance_v):
            break        
    V_out = dev_V_out.copy_to_host()
    policy_out = dev_policy.copy_to_host()
    cuda.synchronize
    t2 = time.time()
    if verbose:
        print(f"JCR PI CONTRACTION CUDA ATOMICMAX DONE. [d_inf: {str(d[0])}, main iterations: {k_main}, evaluation iterations total: {k_eval_total}, time: {t2 - t1} s]")    
    return policy_out, V_out, d[0], k_main, k_eval_total, t2 - t1, history_for_plots

@cuda.jit(void(float32[:]))    
def jcr_pi_contraction_cuda_atomicmax_dreset(d): # called exactly for 1 thread
    d[0] = float32(0.0) 

@cuda.jit(void(int32[:]))    
def jcr_pi_contraction_cuda_atomicmax_psreset(policy_stable): # called exactly for 1 thread
    policy_stable[0] = int32(1) 
                
@cuda.jit(void(float32[:,:,:,:], float32[:], int16[:], float32, float32, float32[:], float32[:]))
def jcr_pi_contraction_cuda_atomicmax_eval(P, V_in, policy, reward_car_moved, gamma, V_out, d):
    const_states = cuda.const.array_like(STATES)
    const_actions = cuda.const.array_like(ACTIONS)
    const_rewards_rental = cuda.const.array_like(REWARDS_RENTAL)
    shared_v_new = cuda.shared.array(512, dtype=float32) # corresponds to DEFAULT_MAX_TPB
    shared_v_in = cuda.shared.array(1024, dtype=float32) # corresponds to MAX_N_STATES
    s_index = cuda.blockIdx.x
    t = cuda.threadIdx.x
    tpb = cuda.blockDim.x
    n_states = const_states.shape[0]
    n_rewards_rental = const_rewards_rental.shape[0]    
    nspt = (n_states + tpb - 1) // tpb # next states per thread
    next_state_index = t
    for _ in range(nspt):
        if next_state_index < n_states:
            shared_v_in[next_state_index] = V_in[next_state_index]
        next_state_index += tpb 
    cuda.syncthreads()
    a_index = policy[s_index]
    reward_cars_moved = math.fabs(const_actions[a_index]) * reward_car_moved
    t_part_sum = float32(0.0)    
    for r_index in range(n_rewards_rental):
        r = reward_cars_moved + const_rewards_rental[r_index]        
        next_state_index = t
        for _ in range(nspt):
            if next_state_index < n_states: 
                t_part_sum += P[s_index, a_index, r_index, next_state_index] * (r + gamma * shared_v_in[next_state_index])             
            next_state_index += tpb
    shared_v_new[t] = t_part_sum
    cuda.syncthreads()
    stride = tpb >> 1
    while stride > 0: # sum-reduction
        if t < stride:
            shared_v_new[t] += shared_v_new[t + stride]
        cuda.syncthreads()
        stride >>= 1    
    if t == 0:
        v_new = shared_v_new[0]
        v = V_in[s_index]
        cuda.atomic.max(d, 0, math.fabs(v - v_new))
        V_out[s_index] = v_new        

@cuda.jit(void(float32[:,:,:,:], float32[:], float32, float32, int16[:], int32[:]))
def jcr_pi_contraction_cuda_atomicmax_improve(P, V, reward_car_moved, gamma, policy, policy_stable):
    const_states = cuda.const.array_like(STATES)
    const_actions = cuda.const.array_like(ACTIONS)
    const_rewards_rental = cuda.const.array_like(REWARDS_RENTAL)
    shared_q_new = cuda.shared.array(512, dtype=float32) # corresponds to DEFAULT_MAX_TPB
    shared_v_in = cuda.shared.array(1024, dtype=float32) # corresponds to MAX_N_STATES
    s_index = cuda.blockIdx.x
    t = cuda.threadIdx.x
    tpb = cuda.blockDim.x
    n_states = const_states.shape[0]
    n_rewards_rental = const_rewards_rental.shape[0]    
    nspt = (n_states + tpb - 1) // tpb # next states per thread
    next_state_index = t
    for _ in range(nspt):
        if next_state_index < n_states:
            shared_v_in[next_state_index] = V[next_state_index]
        next_state_index += tpb 
    cuda.syncthreads()      
    a_so_far = policy[s_index] 
    q_max = -float32(inf)
    a_max_index = int16(-1) 
    for a_index in range(const_actions.shape[0]):        
        reward_cars_moved = math.fabs(const_actions[a_index]) * reward_car_moved
        t_part_sum = float32(0.0)
        for r_index in range(n_rewards_rental):
            r = reward_cars_moved + const_rewards_rental[r_index]        
            next_state_index = t
            for _ in range(nspt):
                if next_state_index < n_states:             
                    t_part_sum += P[s_index, a_index, r_index, next_state_index] * (r + gamma * shared_v_in[next_state_index])                 
                next_state_index += tpb
        shared_q_new[t] = t_part_sum
        cuda.syncthreads()
        stride = tpb >> 1
        while stride > 0: # sum-reduction
            if t < stride:
                shared_q_new[t] += shared_q_new[t + stride]
            cuda.syncthreads()
            stride >>= 1 
        if t == 0: 
            q_a = shared_q_new[0]
            if q_a > q_max: 
                q_max = q_a
                a_max_index = a_index
        cuda.syncthreads()
    if t == 0 and a_so_far != a_max_index:
        policy[s_index] = a_max_index
        cuda.atomic.min(policy_stable, 0, int8(0))
    
def jcr_pi_contraction_cuda_atomicmaxglosten(policy_in, V_in, dev_P,                                             
                                             gamma, eps, tolerance_v,
                                             states=STATES, actions=ACTIONS, rewards_rental=REWARDS_RENTAL, reward_car_moved=REWARD_CAR_MOVED,                                              
                                             lazy_stop_check=DEFAULT_LAZY_STOP_CHECK, tpb=DEFAULT_TPB,
                                             verbose=True, verbose_iters=False, plots=False):
    if verbose:
        print(f"JCR PI CONTRACTION CUDA ATOMICMAXGLOSTEN... [gamma: {gamma}, eps: {eps}, tolerance_v: {tolerance_v}, lazy_stop_check: {lazy_stop_check}, tpb: {tpb}]")
    t1 = time.time()
    history_for_plots = []
    if verbose_iters and plots:
        history_for_plots.append((V_in, policy_in))
        plot_value_and_policy_jcr(V_in, policy_in)                    
    dev_V_in = cuda.to_device(V_in)
    dev_V_out = cuda.to_device(V_in)
    dev_policy = cuda.to_device(policy_in)
    d = np.zeros(1, dtype=np.float32)
    dev_d = cuda.to_device(d)
    dev_policy_stable = cuda.to_device(np.ones(1, dtype=np.int32))
    bpg = states.shape[0]
    if verbose:
        print(f"[bpg: {bpg} (implied by number of states)]")
    k_main = 0
    k_eval_total = 0        
    while True:
        if verbose_iters:
            print(f"---")
            print(f"[main iteration {k_main + 1}:]")
        k_eval = 0
        while True:
            t1_eval = time.time()
            jcr_pi_contraction_cuda_atomicmax_dreset[1, 1](dev_d)
            jcr_pi_contraction_cuda_atomicmaxglosten_eval[bpg, tpb](dev_P, dev_V_in, dev_policy, reward_car_moved, gamma, dev_V_out, dev_d)            
            t2_eval = time.time()
            k_eval += 1
            tmp = dev_V_in # ping-pong trick
            dev_V_in = dev_V_out
            dev_V_out = tmp            
            if k_eval % lazy_stop_check == 0:        
                dev_d.copy_to_host(ary=d)
                cuda.synchronize()
                if verbose_iters: 
                    print(f"[policy evaluation iteration {k_eval} [d_inf: {str(d[0])}, time: {t2_eval - t1_eval} s]]")            
                if d[0] <= eps:
                    k_eval_total += k_eval
                    break
        t1_impr = time.time()
        jcr_pi_contraction_cuda_atomicmax_psreset[1, 1](dev_policy_stable)                    
        jcr_pi_contraction_cuda_atomicmaxglosten_improve[bpg, tpb](dev_P, dev_V_in, reward_car_moved, gamma, dev_policy, dev_policy_stable)            
        policy_stable = dev_policy_stable.copy_to_host()[0]
        policy_stable = True if policy_stable == 1 else False
        cuda.synchronize()
        t2_impr = time.time()
        k_main += 1
        if verbose_iters:
            print(f"[policy improvement [policy_stable: {policy_stable}, d_inf: {str(d[0])}, d_inf <= tolerance_v: {d[0] <= tolerance_v}, time: {t2_impr - t1_impr} s]]")        
            if plots:
                V_now = dev_V_in.copy_to_host()
                policy_now = dev_policy.copy_to_host()
                cuda.synchronize()
                history_for_plots.append((V_now, policy_now))                
                plot_value_and_policy_jcr(V_now, policy_now)
        if policy_stable or (k_eval == 1 and d[0] <= tolerance_v):
            break        
    V_out = dev_V_in.copy_to_host()
    policy_out = dev_policy.copy_to_host()
    cuda.synchronize
    t2 = time.time()
    if verbose:
        print(f"JCR PI CONTRACTION CUDA ATOMICMAXGLOSTEN DONE. [d_inf: {str(d[0])}, main iterations: {k_main}, evaluation iterations total: {k_eval_total}, time: {t2 - t1} s]")    
    return policy_out, V_out, d[0], k_main, k_eval_total, t2 - t1, history_for_plots

@cuda.jit(void(float32[:,:,:,:], float32[:], int16[:], float32, float32, float32[:], float32[:]))
def jcr_pi_contraction_cuda_atomicmaxglosten_eval(P, V_in, policy, reward_car_moved, gamma, V_out, d):
    const_states = cuda.const.array_like(STATES)
    const_actions = cuda.const.array_like(ACTIONS)
    const_rewards_rental = cuda.const.array_like(REWARDS_RENTAL)    
    shared_v_new = cuda.shared.array(512, dtype=float32) # corresponds to DEFAULT_MAX_TPB
    s_index = cuda.blockIdx.x
    t = cuda.threadIdx.x
    tpb = cuda.blockDim.x
    n_states = const_states.shape[0]
    n_rewards_rental = const_rewards_rental.shape[0]    
    nspt = (n_states + tpb - 1) // tpb # next states per thread  
    a_index = policy[s_index]
    reward_cars_moved = math.fabs(const_actions[a_index]) * reward_car_moved
    t_part_sum = float32(0.0)
    for r_index in range(n_rewards_rental):
        r = reward_cars_moved + const_rewards_rental[r_index]        
        next_state_index = t
        for _ in range(nspt):
            if next_state_index < n_states:             
                t_part_sum += P[s_index, a_index, r_index, next_state_index] * (r + gamma * V_in[next_state_index])        
            next_state_index += tpb
    shared_v_new[t] = t_part_sum
    cuda.syncthreads()
    stride = tpb >> 1
    while stride > 0: # sum-reduction
        if t < stride:
            shared_v_new[t] += shared_v_new[t + stride]
        cuda.syncthreads()
        stride >>= 1            
    if t == 0:        
        v = V_in[s_index]
        v_new = shared_v_new[0]
        cuda.atomic.max(d, 0, math.fabs(v - v_new))
        V_out[s_index] = v_new
        
@cuda.jit(void(float32[:,:,:,:], float32[:], float32, float32, int16[:], int32[:]))
def jcr_pi_contraction_cuda_atomicmaxglosten_improve(P, V, reward_car_moved, gamma, policy, policy_stable):
    const_states = cuda.const.array_like(STATES)
    const_actions = cuda.const.array_like(ACTIONS)
    const_rewards_rental = cuda.const.array_like(REWARDS_RENTAL)
    shared_q_new = cuda.shared.array(512, dtype=float32) # corresponds to DEFAULT_MAX_TPBX
    s_index = cuda.blockIdx.x
    t = cuda.threadIdx.x
    tpb = cuda.blockDim.x
    n_states = const_states.shape[0]
    n_rewards_rental = const_rewards_rental.shape[0]    
    nspt = (n_states + tpb - 1) // tpb # next states per thread  
    a_so_far = policy[s_index] 
    q_max = -float32(inf)
    a_max_index = int16(-1) 
    for a_index in range(const_actions.shape[0]):        
        reward_cars_moved = math.fabs(const_actions[a_index]) * reward_car_moved
        t_part_sum = float32(0.0)
        for r_index in range(n_rewards_rental):
            r = reward_cars_moved + const_rewards_rental[r_index]        
            next_state_index = t
            for _ in range(nspt):
                if next_state_index < n_states:             
                    t_part_sum += P[s_index, a_index, r_index, next_state_index] * (r + gamma * V[next_state_index])
                next_state_index += tpb
        shared_q_new[t] = t_part_sum
        cuda.syncthreads()                
        stride = tpb >> 1
        while stride > 0: # sum-reduction
            if t < stride:
                shared_q_new[t] += shared_q_new[t + stride]
            cuda.syncthreads()
            stride >>= 1
        if t == 0: 
            q_a = shared_q_new[0]
            if q_a > q_max:            
                q_max = shared_q_new[0]
                a_max_index = a_index
        cuda.syncthreads()
    if t == 0 and a_so_far != a_max_index:
        policy[s_index] = a_max_index
        cuda.atomic.min(policy_stable, 0, int8(0))        
                
def jcr_pi_contraction_cuda_reducemax(policy_in, V_in, dev_P,                                             
                                      gamma, eps, tolerance_v,
                                      states=STATES, actions=ACTIONS, rewards_rental=REWARDS_RENTAL, reward_car_moved=REWARD_CAR_MOVED,                                              
                                      lazy_stop_check=DEFAULT_LAZY_STOP_CHECK, tpb=DEFAULT_TPB,
                                      verbose=True, verbose_iters=False, plots=False):
    if verbose:
        print(f"JCR PI CONTRACTION CUDA REDUCEMAX... [gamma: {gamma}, eps: {eps}, tolerance_v: {tolerance_v}, lazy_stop_check: {lazy_stop_check}, tpb: {tpb}]")
    t1 = time.time()
    history_for_plots = []
    if verbose_iters and plots:
        history_for_plots.append((V_in, policy_in))
        plot_value_and_policy_jcr(V_in, policy_in)                
    dev_V_in = cuda.to_device(V_in)
    dev_V_out = cuda.to_device(V_in)
    dev_policy = cuda.to_device(policy_in)
    d = np.zeros(states.shape[0], dtype=np.float32)
    dev_d = cuda.to_device(d)
    dev_policy_stable = cuda.to_device(np.ones(states.shape[0], dtype=np.int32))        
    bpg = states.shape[0]
    if verbose:
        print(f"[bpg: {bpg} (implied by number of states)]")
    k_main = 0
    k_eval_total = 0        
    while True:
        if verbose_iters:
            print(f"---")
            print(f"[main iteration {k_main + 1}:]")        
        k_eval = 0
        while True:
            t1_eval = time.time()                   
            jcr_pi_contraction_cuda_reducemax_eval[bpg, tpb](dev_P, dev_V_in, dev_policy, reward_car_moved, gamma, dev_V_out, dev_d)            
            t2_eval = time.time()
            k_eval += 1
            tmp = dev_V_in # ping-pong trick
            dev_V_in = dev_V_out
            dev_V_out = tmp
            if k_eval % lazy_stop_check == 0:
                jcr_pi_contraction_cuda_reducemax_dreduce[1, tpb](dev_d)        
                dev_d.copy_to_host(ary=d)
                cuda.synchronize()
                if verbose_iters: 
                    print(f"[policy evaluation iteration {k_eval} [d_inf: {str(d[0])}, time: {t2_eval - t1_eval} s]]")            
                if d[0] <= eps:
                    k_eval_total += k_eval
                    break
        t1_impr = time.time()            
        jcr_pi_contraction_cuda_reducemax_improve[bpg, tpb](dev_P, dev_V_in, reward_car_moved, gamma, dev_policy, dev_policy_stable)
        jcr_pi_contraction_cuda_reducemax_psreduce[1, tpb](dev_policy_stable)            
        policy_stable = dev_policy_stable.copy_to_host()[0]
        policy_stable = True if policy_stable == 1 else False
        cuda.synchronize()
        t2_impr = time.time()
        k_main += 1
        if verbose_iters:
            print(f"[policy improvement [policy_stable: {policy_stable}, d_inf: {str(d[0])}, d_inf <= tolerance_v: {d[0] <= tolerance_v}, time: {t2_impr - t1_impr} s]]")        
            if plots:                
                V_now = dev_V_in.copy_to_host()
                policy_now = dev_policy.copy_to_host()
                cuda.synchronize()
                history_for_plots.append((V_now, policy_now))                
                plot_value_and_policy_jcr(V_now, policy_now)
        if policy_stable or (k_eval == 1 and d[0] <= tolerance_v):
            break        
    V_out = dev_V_out.copy_to_host()
    policy_out = dev_policy.copy_to_host()
    cuda.synchronize
    t2 = time.time()
    if verbose:
        print(f"JCR PI CONTRACTION CUDA REDUCEMAX DONE. [d_inf: {str(d[0])}, main iterations: {k_main}, evaluation iterations total: {k_eval_total}, time: {t2 - t1} s]")    
    return policy_out, V_out, d[0], k_main, k_eval_total, t2 - t1, history_for_plots                

@cuda.jit(void(float32[:,:,:,:], float32[:], int16[:], float32, float32, float32[:], float32[:]))
def jcr_pi_contraction_cuda_reducemax_eval(P, V_in, policy, reward_car_moved, gamma, V_out, d):
    const_states = cuda.const.array_like(STATES)
    const_actions = cuda.const.array_like(ACTIONS)
    const_rewards_rental = cuda.const.array_like(REWARDS_RENTAL)
    shared_v_new = cuda.shared.array(512, dtype=float32) # corresponds to DEFAULT_MAX_TPB
    shared_v_in = cuda.shared.array(1024, dtype=float32) # corresponds to MAX_N_STATES
    s_index = cuda.blockIdx.x
    t = cuda.threadIdx.x
    tpb = cuda.blockDim.x
    n_states = const_states.shape[0]
    n_rewards_rental = const_rewards_rental.shape[0]    
    nspt = (n_states + tpb - 1) // tpb # next states per thread
    next_state_index = t
    for _ in range(nspt):
        if next_state_index < n_states:
            shared_v_in[next_state_index] = V_in[next_state_index]
        next_state_index += tpb 
    cuda.syncthreads()
    a_index = policy[s_index]
    reward_cars_moved = math.fabs(const_actions[a_index]) * reward_car_moved
    t_part_sum = float32(0.0)    
    for r_index in range(n_rewards_rental):
        r = reward_cars_moved + const_rewards_rental[r_index]        
        next_state_index = t
        for _ in range(nspt):
            if next_state_index < n_states: 
                t_part_sum += P[s_index, a_index, r_index, next_state_index] * (r + gamma * shared_v_in[next_state_index])             
            next_state_index += tpb
    shared_v_new[t] = t_part_sum
    cuda.syncthreads()
    stride = tpb >> 1
    while stride > 0: # sum-reduction
        if t < stride:
            shared_v_new[t] += shared_v_new[t + stride]
        cuda.syncthreads()
        stride >>= 1    
    if t == 0:
        v_new = shared_v_new[0]
        v = V_in[s_index]
        d[s_index] = math.fabs(v - v_new)        
        V_out[s_index] = v_new

@cuda.jit(void(float32[:]))    
def jcr_pi_contraction_cuda_reducemax_dreduce(d):         
    shared_d = cuda.shared.array(2048, dtype=float32) # corresponds to MAX_N_STATES
    tpb = cuda.blockDim.x
    job_blocks = d.shape[0]
    ept = (job_blocks + tpb - 1) // tpb 
    t = cuda.threadIdx.x
    e = t
    shared_d[t] = float32(0.0)
    for _ in range(ept):
        if e < job_blocks:
            shared_d[t] = max(shared_d[t], d[e])
        e += tpb    
    cuda.syncthreads()
    stride = tpb >> 1       
    cuda.syncthreads()
    while stride > 0: # max-reduction        
        if t < stride:
            shared_d[t] = max(shared_d[t], shared_d[t + stride])
        cuda.syncthreads()
        stride >>= 1
    if t == 0:    
        d[0] = shared_d[0]

@cuda.jit(void(float32[:,:,:,:], float32[:], float32, float32, int16[:], int32[:]))
def jcr_pi_contraction_cuda_reducemax_improve(P, V, reward_car_moved, gamma, policy, policy_stable):
    const_states = cuda.const.array_like(STATES)
    const_actions = cuda.const.array_like(ACTIONS)
    const_rewards_rental = cuda.const.array_like(REWARDS_RENTAL)
    shared_q_new = cuda.shared.array(512, dtype=float32) # corresponds to DEFAULT_MAX_TPB
    shared_v_in = cuda.shared.array(1024, dtype=float32) # corresponds to MAX_N_STATES
    s_index = cuda.blockIdx.x
    t = cuda.threadIdx.x
    tpb = cuda.blockDim.x
    n_states = const_states.shape[0]
    n_rewards_rental = const_rewards_rental.shape[0]    
    nspt = (n_states + tpb - 1) // tpb # next states per thread
    next_state_index = t
    for _ in range(nspt):
        if next_state_index < n_states:
            shared_v_in[next_state_index] = V[next_state_index]
        next_state_index += tpb 
    cuda.syncthreads()      
    a_so_far = policy[s_index] 
    q_max = -float32(inf)
    a_max_index = int16(-1) 
    for a_index in range(const_actions.shape[0]):
        reward_cars_moved = math.fabs(const_actions[a_index]) * reward_car_moved
        t_part_sum = float32(0.0)
        for r_index in range(n_rewards_rental):
            r = reward_cars_moved + const_rewards_rental[r_index]        
            next_state_index = t
            for _ in range(nspt):
                if next_state_index < n_states:             
                    t_part_sum += P[s_index, a_index, r_index, next_state_index] * (r + gamma * shared_v_in[next_state_index])                 
                next_state_index += tpb
        shared_q_new[t] = t_part_sum    
        cuda.syncthreads()
        stride = tpb >> 1
        while stride > 0: # sum-reduction
            if t < stride:
                shared_q_new[t] += shared_q_new[t + stride]
            cuda.syncthreads()
            stride >>= 1     
        if t == 0: 
            q_a = shared_q_new[0] 
            if q_a > q_max:
                q_max = q_a
                a_max_index = a_index
        cuda.syncthreads()
    if t == 0:
        if a_so_far != a_max_index:
            policy[s_index] = a_max_index
        policy_stable[s_index] = int32(0) if a_so_far != a_max_index else int32(1)

@cuda.jit(void(int32[:]))    
def jcr_pi_contraction_cuda_reducemax_psreduce(policy_stable):         
    shared_policy_stable = cuda.shared.array(2048, dtype=int32) # corresponds to MAX_N_STATES
    tpb = cuda.blockDim.x
    job_blocks = policy_stable.shape[0]
    ept = (job_blocks + tpb - 1) // tpb 
    t = cuda.threadIdx.x
    e = t
    shared_policy_stable[t] = int32(1)
    for _ in range(ept):
        if e < job_blocks:
            shared_policy_stable[t] = min(shared_policy_stable[t], policy_stable[e])
        e += tpb    
    cuda.syncthreads()
    stride = tpb >> 1       
    cuda.syncthreads()
    while stride > 0: # max-reduction        
        if t < stride:
            shared_policy_stable[t] = min(shared_policy_stable[t], shared_policy_stable[t + stride])
        cuda.syncthreads()
        stride >>= 1
    if t == 0:    
        policy_stable[0] = shared_policy_stable[0]

def jcr_pi_contraction_cuda_gridsync(policy_in, V_in, dev_P,                                             
                                     gamma, eps, tolerance_v,
                                     states=STATES, actions=ACTIONS, rewards_rental=REWARDS_RENTAL, reward_car_moved=REWARD_CAR_MOVED,                                              
                                     tpb=DEFAULT_TPB, max_bpg_gridsync=None,
                                     verbose=True, verbose_iters=False, plots=False):    
    if verbose:
        print(f"JCR PI CONTRACTION CUDA GRIDSYNC... [gamma: {gamma}, eps: {eps}, tolerance_v: {tolerance_v}, tpb: {tpb}, assumed max_bpg_gridsync: {max_bpg_gridsync}]")
    t1 = time.time()    
    history_for_plots = []
    if verbose_iters and plots:
        history_for_plots.append((V_in, policy_in))
        plot_value_and_policy_jcr(V_in, policy_in)                
    dev_V_in = cuda.to_device(V_in)
    dev_V_out = cuda.to_device(V_in)
    dev_policy = cuda.to_device(policy_in)
    d = np.zeros(states.shape[0], dtype=np.float32)
    dev_d = cuda.to_device(d)
    dev_policy_stable = cuda.to_device(np.ones(1, dtype=np.int32))
    dev_k_eval_total = cuda.to_device(np.zeros(1, dtype=np.int32))
    dev_stop_all = cuda.to_device(np.zeros(1, dtype=bool))
    if max_bpg_gridsync is None:
        t1_discover = time.time()        
        compiled = jcr_pi_contraction_cuda_gridsync_eval.overloads[(float32[:,:,:,:], float32[:], int16[:], float32, float32, float32, float32[:], float32[:], int32[:], boolean[:],)]
        max_bpg_gridsync = compiled.max_cooperative_grid_blocks(tpb)
        t2_discover = time.time()
        if verbose:
            print(f"[discovered max_bpg_gridsync: {max_bpg_gridsync}; time: {t2_discover - t1_discover} s]")
    bpg_eval = min(max_bpg_gridsync, states.shape[0])
    bpg_improve = states.shape[0]
    if verbose:
        spb = (states.shape[0] + bpg_eval - 1) // bpg_eval
        print(f"[kernel eval -> bpg: {bpg_eval} (max_bpg_gridsync or number of states if smaller), tpb: {tpb}, spb: {spb}]")
        print(f"[kernel improve -> bpg: {bpg_improve} (implied by number of states), tpb: {tpb}]")
    k_main = 0       
    k_eval_total_so_far = 0
    while True:
        if verbose_iters:
            print(f"---")
            print(f"[main iteration {k_main + 1}:]")        
        t1_eval = time.time()
        jcr_pi_contraction_cuda_gridsync_reset[1, 1](dev_d, dev_stop_all)
        jcr_pi_contraction_cuda_gridsync_eval[bpg_eval, tpb](dev_P, dev_V_in, dev_policy, reward_car_moved, gamma, eps, dev_V_out, dev_d, dev_k_eval_total, dev_stop_all)                        
        k_eval_total = dev_k_eval_total.copy_to_host()[0]
        cuda.synchronize()
        k_eval = k_eval_total - k_eval_total_so_far
        k_eval_total_so_far = k_eval_total
        if k_eval % 2 == 1:            
            dev_V_in, dev_V_out = dev_V_out, dev_V_in        
        t2_eval = time.time()
        if verbose_iters:             
            d = dev_d.copy_to_host()
            cuda.synchronize()
            print(f"[policy evaluation iterations {k_eval} [d_inf: {str(d[0])}, time: {t2_eval - t1_eval} s]]")
        t1_impr = time.time()            
        jcr_pi_contraction_cuda_gridsync_psreset[1, 1](dev_policy_stable)                   
        jcr_pi_contraction_cuda_gridsync_improve[bpg_improve, tpb](dev_P, dev_V_in, reward_car_moved, gamma, dev_policy, dev_policy_stable)                            
        policy_stable = dev_policy_stable.copy_to_host()[0]
        policy_stable = True if policy_stable == 1 else False
        cuda.synchronize()
        t2_impr = time.time()
        k_main += 1
        if verbose_iters:
            print(f"[policy improvement [policy_stable: {policy_stable}, d_inf: {str(d[0])}, d_inf <= tolerance_v: {d[0] <= tolerance_v}, time: {t2_impr - t1_impr} s]]")        
            if plots:                
                V_now = dev_V_in.copy_to_host()
                policy_now = dev_policy.copy_to_host()
                cuda.synchronize()
                history_for_plots.append((V_now, policy_now))                
                plot_value_and_policy_jcr(V_now, policy_now)
        if policy_stable or (k_eval == 1 and d[0] <= tolerance_v):
            break        
    V_out = dev_V_out.copy_to_host()
    policy_out = dev_policy.copy_to_host()
    d = dev_d.copy_to_host()
    cuda.synchronize
    t2 = time.time()
    if verbose:
        print(f"JCR PI CONTRACTION CUDA GRIDSYNC DONE. [d_inf: {str(d[0])}, main iterations: {k_main}, evaluation iterations total: {k_eval_total}, time: {t2 - t1} s]")    
    return policy_out, V_out, d[0], k_main, k_eval_total_so_far, t2 - t1, history_for_plots

@cuda.jit(void(float32[:], boolean[:]))    
def jcr_pi_contraction_cuda_gridsync_reset(d, stop_all): # called exactly for 1 thread
    d[0] = float32(0.0)
    stop_all[0] = False

@cuda.jit(void(float32[:,:,:,:], float32[:], int16[:], float32, float32, float32, float32[:], float32[:], int32[:], boolean[:]))
def jcr_pi_contraction_cuda_gridsync_eval(P, V_in, policy, reward_car_moved, gamma, eps, V_out, d, k_eval_total, stop_all):
    const_states = cuda.const.array_like(STATES)
    const_actions = cuda.const.array_like(ACTIONS)
    const_rewards_rental = cuda.const.array_like(REWARDS_RENTAL)    
    shared_v_new = cuda.shared.array(512, dtype=float32) # corresponds to DEFAULT_MAX_TPB
    shared_v_in = cuda.shared.array(1024, dtype=float32) # corresponds to MAX_N_STATES
    bpg = cuda.gridDim.x
    n_states = const_states.shape[0]
    spb = (n_states + bpg - 1) // bpg # states per block
    g = cuda.cg.this_grid()
    b = cuda.blockIdx.x     
    t = cuda.threadIdx.x
    tpb = cuda.blockDim.x    
    n_rewards_rental = const_rewards_rental.shape[0]    
    nspt = (n_states + tpb - 1) // tpb # next states per thread
    k = int32(0)
    V_src = V_in
    V_dst = V_out
    while True:
        next_state_index = t
        for _ in range(nspt):
            if next_state_index < n_states:
                shared_v_in[next_state_index] = V_src[next_state_index]
            next_state_index += tpb 
        cuda.syncthreads()
        s_index = cuda.blockIdx.x    
        for _ in range(spb):
            if s_index < n_states:        
                a_index = policy[s_index]
                reward_cars_moved = math.fabs(const_actions[a_index]) * reward_car_moved
                t_part_sum = float32(0.0)
                for r_index in range(n_rewards_rental):
                    r = reward_cars_moved + const_rewards_rental[r_index]        
                    next_state_index = t
                    for _ in range(nspt):
                        if next_state_index < n_states:             
                            t_part_sum += P[s_index, a_index, r_index, next_state_index] * (r + gamma * shared_v_in[next_state_index])        
                        next_state_index += tpb
                shared_v_new[t] = t_part_sum
                cuda.syncthreads()
                stride = tpb >> 1
                while stride > 0: # sum-reduction
                    if t < stride:
                        shared_v_new[t] += shared_v_new[t + stride]
                    cuda.syncthreads()
                    stride >>= 1            
                if t == 0:        
                    v = V_src[s_index]
                    v_new = shared_v_new[0]
                    cuda.atomic.max(d, 0, math.fabs(v - v_new))
                    V_dst[s_index] = v_new
            s_index += bpg
        k += 1
        g.sync()
        if b == 0 and t == 0:
            if d[0] <= eps:                
                stop_all[0] = True
                k_eval_total[0] += k        
            else:
                d[0] = float32(0.0)            
        g.sync()
        if stop_all[0]:            
            break
        tmp = V_src # ping-poing trick
        V_src = V_dst
        V_dst = tmp
        
@cuda.jit(void(int32[:]))    
def jcr_pi_contraction_cuda_gridsync_psreset(policy_stable): # called exactly for 1 thread
    policy_stable[0] = int32(1) 
        
@cuda.jit(void(float32[:,:,:,:], float32[:], float32, float32, int16[:], int32[:]))
def jcr_pi_contraction_cuda_gridsync_improve(P, V, reward_car_moved, gamma, policy, policy_stable):
    const_states = cuda.const.array_like(STATES)
    const_actions = cuda.const.array_like(ACTIONS)
    const_rewards_rental = cuda.const.array_like(REWARDS_RENTAL)
    shared_q_new = cuda.shared.array(512, dtype=float32) # corresponds to DEFAULT_MAX_TPB
    shared_v_in = cuda.shared.array(1024, dtype=float32) # corresponds to MAX_N_STATES
    s_index = cuda.blockIdx.x
    t = cuda.threadIdx.x
    tpb = cuda.blockDim.x
    n_states = const_states.shape[0]
    n_rewards_rental = const_rewards_rental.shape[0]    
    nspt = (n_states + tpb - 1) // tpb # next states per thread
    next_state_index = t
    for _ in range(nspt):
        if next_state_index < n_states:
            shared_v_in[next_state_index] = V[next_state_index]
        next_state_index += tpb 
    cuda.syncthreads()      
    a_so_far = policy[s_index] 
    q_max = -float32(inf)
    a_max_index = int16(-1) 
    for a_index in range(const_actions.shape[0]):        
        reward_cars_moved = math.fabs(const_actions[a_index]) * reward_car_moved
        t_part_sum = float32(0.0)
        for r_index in range(n_rewards_rental):
            r = reward_cars_moved + const_rewards_rental[r_index]        
            next_state_index = t
            for _ in range(nspt):
                if next_state_index < n_states:             
                    t_part_sum += P[s_index, a_index, r_index, next_state_index] * (r + gamma * shared_v_in[next_state_index])                 
                next_state_index += tpb
        shared_q_new[t] = t_part_sum
        cuda.syncthreads()
        stride = tpb >> 1
        while stride > 0: # sum-reduction
            if t < stride:
                shared_q_new[t] += shared_q_new[t + stride]
            cuda.syncthreads()
            stride >>= 1 
        if t == 0: 
            q_a = shared_q_new[0]
            if q_a > q_max: 
                q_max = q_a
                a_max_index = a_index
        cuda.syncthreads()
    if t == 0 and a_so_far != a_max_index:
        policy[s_index] = a_max_index
        cuda.atomic.min(policy_stable, 0, int8(0))

def plot_value_and_policy_jcr(V, policy, states=STATES, actions=ACTIONS, V_plot_on=True, policy_plot_on=True, plot_titles_on=True, cmap_for_V_plot=False, overall_title=None):    
    n_plots = sum([V_plot_on, policy_plot_on])    
    if n_plots == 0:
        print("BOTH PLOTS ARE OFF.")
        return
    figsize = (6 * n_plots, 6)
    fig = plt.figure(figsize=figsize)
    fig.canvas.manager.set_window_title("JACK'S CAR RENTAL")
    if overall_title:
        fig.suptitle(overall_title, fontsize=12, y=0.985)    
    max_cars_at_loc = int(np.sqrt(states.shape[0])) - 1
    max_cars_moved = (actions.shape[0] - 1) // 2    
    current_ax_idx = 1 
    if V_plot_on:
        ax1 = fig.add_subplot(1, n_plots, current_ax_idx, projection="3d")
        V_to_plot = V.reshape((max_cars_at_loc + 1, max_cars_at_loc + 1))
        X, Y = np.meshgrid(np.arange(V_to_plot.shape[0]), np.arange(V_to_plot.shape[1]))        
        if cmap_for_V_plot:
            ax1.plot_surface(X, Y, V_to_plot, cmap="viridis", edgecolor="k", linewidth=0.5)
        else:
            ax1.plot_surface(X, Y, V_to_plot, color="white", edgecolor="k", linewidth=0.5, alpha=0.5)        
        ax1.set_xlabel("CARS AT LOCATION 2")
        ax1.set_ylabel("CARS AT LOCATION 1")
        ax1.set_yticks(np.arange(0, max_cars_at_loc + 1, 5)) 
        ax1.set_xticks(np.arange(0, max_cars_at_loc + 1, 5))
        if plot_titles_on: 
            ax1.set_title("VALUE FUNCTION (V)")        
        current_ax_idx += 1
    if policy_plot_on:
        ax2 = fig.add_subplot(1, n_plots, current_ax_idx)
        policy_to_plot = actions[policy].reshape((max_cars_at_loc + 1, max_cars_at_loc + 1))
        extent = [-0.5, max_cars_at_loc + 0.5, -0.5, max_cars_at_loc + 0.5]        
        ax2.imshow(policy_to_plot, cmap="coolwarm", origin="lower", 
                   vmin=-max_cars_moved, vmax=max_cars_moved, extent=extent)        
        for act in range(-max_cars_moved, max_cars_moved + 1):
            action_mask = (policy_to_plot == act)
            labeled_array, num_features = label(action_mask)
            for feature_id in range(1, num_features + 1):
                coords = np.argwhere(labeled_array == feature_id)
                if coords.size > 0:
                    y_mid, x_mid = np.median(coords, axis=0)
                    ax2.text(x_mid, y_mid, f"{act:+d}" if act != 0 else "0", 
                             color="black", fontsize=8, fontweight="bold", 
                             ha="center", va="center")        
        ax2.set_xlabel("CARS AT LOCATION 2")
        ax2.set_ylabel("CARS AT LOCATION 1")
        ax2.set_yticks(np.arange(0, max_cars_at_loc + 1, 5)) 
        ax2.set_xticks(np.arange(0, max_cars_at_loc + 1, 5))
        if plot_titles_on: 
            ax2.set_title("POLICY")
    plt.show()
