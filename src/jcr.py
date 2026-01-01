import numpy as np
from numpy import inf
import math
import itertools
import time
from numba import cuda
from numba import void, int8, int16, int32, float32, boolean
from numba.core.errors import NumbaPerformanceWarning
import math
import warnings
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
import os
os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"] = "1"
from matplotlib import pyplot as plt
from scipy.ndimage import label

# global settings
MAX_CARS_AT_LOC = 20
MAX_CARS_MOVED = 5
REWARD_CAR_RENTED = 10.0
REWARD_CAR_MOVED = -2.0
GAMMA = 0.9
EPS = 1e-2
TOLERANCE_V = 1e-7
STATES = np.array(list(itertools.product(np.arange(0, MAX_CARS_AT_LOC + 1), np.arange(0, MAX_CARS_AT_LOC + 1))), dtype=np.int16)
ACTIONS = np.arange(-MAX_CARS_MOVED, MAX_CARS_MOVED + 1, dtype=np.int16)
REWARDS = np.arange(2 * MAX_CARS_AT_LOC + 1) * REWARD_CAR_RENTED
LAMBDAS = [2.0, 3.0, 4.0]
MAX_N = MAX_CARS_AT_LOC
POISSON_DISTRS = {l: np.array([l**n * np.exp(-l) / math.factorial(n) for n in range(MAX_N + 1)]) for l in LAMBDAS}
for k in POISSON_DISTRS.keys():
    distr = POISSON_DISTRS[k]
    distr[-1] = 1.0 - np.sum(distr[:-1])
REQUEST_DISTRS_AT_LOCS = [POISSON_DISTRS[3.0], POISSON_DISTRS[4.0]]
RETURN_DISTRS_AT_LOCS = [POISSON_DISTRS[3.0], POISSON_DISTRS[2.0]]
MDP_PATH = f"../mdp_joint_distrs/mdp_joint_distr_jcr_{MAX_CARS_AT_LOC}_{MAX_CARS_MOVED}.pkl"

# CUDA defaults
DEFAULT_TPB = cuda.get_current_device().MAX_THREADS_PER_BLOCK // 2 
DEFAULT_LAZY_STOP_CHECK = 100
DEFAULT_GRIDSYNC_MAX_BPG = 100
DEFAULT_CORES = 1024

# constraints
MAX_N_STATES = 2048
assert (MAX_CARS_AT_LOC + 1)**2 <= MAX_N_STATES, f"Maximum number of states {MAX_N_STATES} exceeded due to too large maximum of cars at location set to: {MAX_CARS_AT_LOC}."

def mdp_joint_distr_jcr(states=STATES, actions=ACTIONS, rewards=REWARDS, request_distrs_at_locs=REQUEST_DISTRS_AT_LOCS, return_distrs_at_locs=RETURN_DISTRS_AT_LOCS, 
                        return_as_float32_array=True):
    print("MDP_JOINT_DISTR_JCR...")
    t1 = time.time()
    max_cars_at_loc = int(np.sqrt(states.shape[0])) - 1
    P = np.zeros((rewards.shape[0], states.shape[0], states.shape[0], actions.shape[0]))
    print(f"[shape of P to prepare: {P.shape}, entries: {P.size}]")
    for s_index in range(states.shape[0]):
        print(f"PROGRESS: {s_index + 1}/{states.shape[0]}.")
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
                    P[reward_index, s_next_index, s_index, a_index] += P_request * P_return
    if return_as_float32_array:
        P = P.astype(np.float32)                                                             
    t2 = time.time()
    print(f"MDP_JOINT_DISTR_JCR DONE. [time: {t2 - t1} s]")
    return P

def jcr_pi_contraction_numpy(P, states=STATES, actions=ACTIONS, rewards=REWARDS, reward_car_moved=REWARD_CAR_MOVED, gamma=GAMMA, eps=EPS, tolerance_v=TOLERANCE_V, seed=0, plots=False): # TODO rename rewards to rewards_rental
    print("JCR PI CONTRACTION NUMPY...")
    t1 = time.time()
    max_cars_at_loc = int(np.sqrt(states.shape[0])) - 1
    max_cars_moved = (actions.shape[0] - 1) // 2    
    # initialization
    V = np.zeros(states.shape[0], dtype=P.dtype)
    # policy = np.zeros(STATES.shape[0], dtype=np.int16) + MAX_CARS_MOVED # corresponds to action: move 0 cars (hence the shift)
    np.random.seed(seed)     
    policy = np.random.randint(-max_cars_moved, max_cars_moved + 1, size=states.shape[0], dtype=np.int16) + max_cars_moved
    print(actions[policy].reshape((max_cars_at_loc + 1, max_cars_at_loc + 1))) # TODO remove later
    if plots:
        print(actions[policy].reshape((max_cars_at_loc + 1, max_cars_at_loc + 1)))
        plot_value_and_policy_jcr(V, policy)    
    i_main = 0 
    while True:
        print(f"---")        
        print(f"MAIN ITERATION {i_main + 1}:")
        # POLICY EVALUATION        
        i_eval = 0
        while True:
            t1_eval = time.time()    
            d = 0                     
            for s_index in range(states.shape[0]):
                v = V[s_index]
                a_index = policy[s_index]
                reward_cars_moved = np.abs(actions[a_index]) * reward_car_moved
                v_new = 0.0
                for r_index in range(rewards.shape[0]):
                    r = reward_cars_moved + rewards[r_index]
                    v_new += np.sum(P[r_index, :, s_index, a_index] * (r + gamma * V))
                V[s_index] = v_new
                d = max(d, np.abs(v - v_new))
            i_eval += 1
            t2_eval = time.time()
            print(f"POLICY EVALUATION ITERATION {i_eval} [d_inf: {d}, time: {t2_eval - t1_eval} s].")
            if d <= eps:
                break
        # POLICY IMPROVEMENT
        t1_impr = time.time()
        policy_stable = True
        for s_index in range(states.shape[0]):
            a_so_far = policy[s_index]
            qs = np.zeros(actions.shape[0])
            for a_index in range(actions.shape[0]):
                reward_cars_moved = np.abs(actions[a_index]) * reward_car_moved
                q = 0.0
                for r_index in range(rewards.shape[0]):
                    r = reward_cars_moved + rewards[r_index]
                    q += np.sum(P[r_index, :, s_index, a_index] * (r + gamma * V))
                qs[a_index] = q
            policy[s_index] = np.argmax(qs)
            if policy[s_index] != a_so_far:
                policy_stable = False 
        t2_impr = time.time()
        print(f"POLICY IMPROVEMENT [policy_stable: {policy_stable}, d_inf: {d}, d_inf < tolerance_v: {d < tolerance_v}, time: {t2_impr - t1_impr} s].")
        if plots:
            print(actions[policy].reshape((max_cars_at_loc + 1, max_cars_at_loc + 1)))
            plot_value_and_policy_jcr(V, policy)
        if policy_stable or (i_eval == 1 and d <= tolerance_v):
            break
        i_main += 1
    t2 = time.time()        
    print(f"JCR PI CONTRACTION NUMPY DONE. [time: {t2 - t1} s, main loop iterations: {i_main}]")
    print(actions[policy].reshape((max_cars_at_loc + 1, max_cars_at_loc + 1))) # TODO remove later    
    return V, policy                                     

def plot_value_and_policy_jcr(V, policy, states=STATES, actions=ACTIONS):
    figsize = (12, 6)    
    fig = plt.figure(figsize=figsize)
    max_cars_at_loc = int(np.sqrt(states.shape[0])) - 1
    max_cars_moved = (actions.shape[0] - 1) // 2   
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    V_to_plot = V.reshape((max_cars_at_loc + 1, max_cars_at_loc + 1))
    X, Y = np.meshgrid(np.arange(V_to_plot.shape[0]), np.arange(V_to_plot.shape[1]))
    ax1.plot_surface(X, Y, V_to_plot, cmap="viridis", edgecolor="k", linewidth=0.5)
    ax1.set_xlabel("CARS AT LOCATION 2")
    ax1.set_ylabel("CARS AT LOCATION 1")
    ax1.set_zlabel("VALUE")
    ax1.set_title("VALUE FUNCTION (V)")    
    ax2 = fig.add_subplot(1, 2, 2)
    policy_to_plot = actions[policy].reshape((max_cars_at_loc + 1, max_cars_at_loc + 1))    
    extent = [-0.5, max_cars_at_loc + 0.5, -0.5, max_cars_at_loc + 0.5]
    ax2.imshow(policy_to_plot, cmap="coolwarm", origin="lower", 
               vmin=-max_cars_moved, vmax=max_cars_moved, extent=extent)       
    for act in range(-max_cars_moved, max_cars_moved + 1):
            action_mask = (policy_to_plot == act)        
            labeled_array, num_features = label(action_mask) # connected components (islands with same value)            
            for feature_id in range(1, num_features + 1):
                coords = np.argwhere(labeled_array == feature_id)                
                if coords.size > 0:
                    # y_mid, x_mid = coords.mean(axis=0)
                    y_mid, x_mid = np.median(coords, axis=0)        
                    ax2.text(x_mid, y_mid, f"{act:+d}" if act != 0 else "0", 
                             color="black", fontsize=8, fontweight="bold", 
                             ha="center", va="center",
                             bbox=dict(facecolor="white", alpha=0.0, edgecolor="none", boxstyle="round, pad=0.1"))    
    ax2.set_xlabel("CARS AT LOCATION 2")
    ax2.set_ylabel("CARS AT LOCATION 1")
    ax2.set_yticks(np.arange(0, max_cars_at_loc + 1, 5)) 
    ax2.set_xticks(np.arange(0, max_cars_at_loc + 1, 5))        
    ax2.set_title("POLICY")
    plt.subplots_adjust(wspace=0.5)              
    plt.show()
    
def jcr_pi_contraction_cuda_atomicmaxglosten(P, states=STATES, actions=ACTIONS, rewards=REWARDS, reward_car_moved=REWARD_CAR_MOVED, gamma=GAMMA, eps=EPS, tolerance_v=TOLERANCE_V, seed=0, plots=False,
                                             lazy_stop_check=DEFAULT_LAZY_STOP_CHECK, tpb=DEFAULT_TPB, verbose=True):
    if verbose:
        print(f"JCR PI CONTRACTION CUDA ATOMICMAXGLOSTEN... [eps: {eps}, lazy_stop_check: {lazy_stop_check}, tpb: {tpb}]")
    t1 = time.time()
    max_cars_at_loc = int(np.sqrt(states.shape[0])) - 1
    max_cars_moved = (actions.shape[0] - 1) // 2
    V = np.zeros(states.shape[0], dtype=P.dtype)
    # policy = np.zeros(STATES.shape[0], dtype=np.int16) + MAX_CARS_MOVED # corresponds to action: move 0 cars (hence the shift)
    np.random.seed(seed)     
    policy = np.random.randint(-max_cars_moved, max_cars_moved + 1, size=states.shape[0], dtype=np.int16) + max_cars_moved
    print(actions[policy].reshape((max_cars_at_loc + 1, max_cars_at_loc + 1))) # TODO remove later    
    if plots:
        print(actions[policy].reshape((max_cars_at_loc + 1, max_cars_at_loc + 1)))
        plot_value_and_policy_jcr(V, policy)            
    i_main = 0
    t1_copy_p = time.time()
    dev_P = cuda.to_device(P)
    cuda.synchronize() # is it needed to know correct time of copying?
    t2_copy_p = time.time()
    if verbose:
        print(f"[copying MDP joint distribution done; time: {t2_copy_p - t1_copy_p} s]")
    t1_copy = time.time()
    dev_V_in = cuda.to_device(V)
    dev_V_out = cuda.to_device(V)
    dev_policy = cuda.to_device(policy)
    cuda.synchronize() # is it needed to know correct time of copying?
    t2_copy = time.time()
    if verbose:
        print(f"[copying V and policy done; time: {t2_copy - t1_copy} s]")
    i_eval_total = 0        
    bpg = states.shape[0]
    if verbose:
        print(f"[bpg: {bpg}]")
    while True:
        print(f"---")        
        print(f"MAIN ITERATION {i_main + 1}:")
        # POLICY EVALUATION        
        i_eval = 0
        while True:
            t1_eval = time.time()
            d = np.zeros(1, dtype=np.float32)
            dev_d = cuda.to_device(d)            
            jcr_pi_contraction_cuda_atomicmax_eval[bpg, tpb](dev_P, dev_V_in, dev_V_out, dev_policy, reward_car_moved, gamma, dev_d)            
            d = dev_d.copy_to_host()[0]
            cuda.synchronize()
            t2_eval = time.time()
            i_eval += 1
            tmp = dev_V_in
            dev_V_in = dev_V_out
            dev_V_out = tmp                        
            print(f"POLICY EVALUATION ITERATION {i_eval} [d_inf: {d}, time: {t2_eval - t1_eval} s].")            
            if d <= eps:
                i_eval_total += i_eval
                break

        t1_impr = time.time()
        policy_stable = np.ones(1, dtype=np.int32)
        dev_policy_stable = cuda.to_device(policy_stable)            
        jcr_pi_contraction_cuda_atomicmax_improve[bpg, tpb](dev_P, dev_V_in, dev_policy, reward_car_moved, gamma, dev_policy_stable)            
        policy_stable = dev_policy_stable.copy_to_host()[0]
        policy_stable = True if policy_stable == 1 else False
        cuda.synchronize()
        t2_impr = time.time()
        print(f"POLICY IMPROVEMENT [policy_stable: {policy_stable}, d_inf: {d}, d_inf < tolerance_v: {d < tolerance_v}, time: {t2_impr - t1_impr} s].")        
        
        if plots:
            print(actions[policy].reshape((max_cars_at_loc + 1, max_cars_at_loc + 1)))
            plot_value_and_policy_jcr(V, policy)
        if policy_stable or (i_eval == 1 and d <= tolerance_v):
            break
        i_main += 1
    V = dev_V_out.copy_to_host()
    policy = dev_policy.copy_to_host()
    cuda.synchronize
    t2 = time.time()
    if verbose:
        print(f"JCR PI CONTRACTION CUDA ATOMICMAXGLOSTEN DONE. [main iterations: {i_main}, evaluation iterations total: {i_eval_total}, time: {t2 - t1} s]")
    print(actions[policy].reshape((max_cars_at_loc + 1, max_cars_at_loc + 1))) # TODO remove later    
    return V, policy, i_eval_total, t2 - t1

@cuda.jit(void(float32[:,:,:,:], float32[:], float32[:], int16[:], float32, float32, float32[:]))
def jcr_pi_contraction_cuda_atomicmaxglosten_eval(P, V_in, V_out, policy, reward_car_moved, gamma, d):
    const_states = cuda.const.array_like(STATES)
    const_actions = cuda.const.array_like(ACTIONS)
    const_rewards = cuda.const.array_like(REWARDS)
    shared_v_new = cuda.shared.array(512, dtype=float32) # corresponds to DEFAULT_TPB
    s_index = cuda.blockIdx.x
    t = cuda.threadIdx.x
    tpb = cuda.blockDim.x
    n_states = const_states.shape[0]
    n_rewards = const_rewards.shape[0]    
    nspt = (n_states + tpb - 1) // tpb # next states per thread  
    a_index = policy[s_index]
    reward_cars_moved = math.fabs(const_actions[a_index]) * reward_car_moved
    v_new = float32(0.0)    
    for r_index in range(n_rewards):
        shared_v_new[t] = float32(0.0)
        r = reward_cars_moved + const_rewards[r_index]        
        next_state_index = t
        for _ in range(nspt):
            if next_state_index < n_states:            
                shared_v_new[t] += P[r_index, next_state_index, s_index, a_index] * (r + gamma * V_in[next_state_index]) 
            cuda.syncthreads()
            next_state_index += tpb                        
        stride = tpb >> 1
        while stride > 0: # sum-reduction
            if t < stride:
                shared_v_new[t] += shared_v_new[t + stride]
            cuda.syncthreads()
            stride >>= 1
        v_new += shared_v_new[0]
        cuda.syncthreads() 
    if t == 0:
        v = V_in[s_index]
        cuda.atomic.max(d, 0, math.fabs(v - v_new))
        V_out[s_index] = v_new               

@cuda.jit(void(float32[:,:,:,:], float32[:], int16[:], float32, float32, int32[:]))
def jcr_pi_contraction_cuda_atomicmaxglosten_improve(P, V, policy, reward_car_moved, gamma, policy_stable):
    const_states = cuda.const.array_like(STATES)
    const_actions = cuda.const.array_like(ACTIONS)
    const_rewards = cuda.const.array_like(REWARDS)
    shared_q_new = cuda.shared.array(512, dtype=float32) # corresponds to DEFAULT_TPB
    s_index = cuda.blockIdx.x
    t = cuda.threadIdx.x
    tpb = cuda.blockDim.x
    n_states = const_states.shape[0]
    n_rewards = const_rewards.shape[0]    
    nspt = (n_states + tpb - 1) // tpb # next states per thread  
    a_so_far = policy[s_index] 
    q_max = -float32(inf)
    a_max_index = int16(-1) 
    for a_index in range(const_actions.shape[0]):
        q_a = float32(0.0)
        reward_cars_moved = math.fabs(const_actions[a_index]) * reward_car_moved
        for r_index in range(n_rewards):
            shared_q_new[t] = float32(0.0)
            r = reward_cars_moved + const_rewards[r_index]        
            next_state_index = t
            for _ in range(nspt):
                if next_state_index < n_states:            
                    shared_q_new[t] += P[r_index, next_state_index, s_index, a_index] * (r + gamma * V[next_state_index]) 
                cuda.syncthreads()
                next_state_index += tpb                        
            stride = tpb >> 1
            while stride > 0: # sum-reduction
                if t < stride:
                    shared_q_new[t] += shared_q_new[t + stride]
                cuda.syncthreads()
                stride >>= 1
            q_a += shared_q_new[0]
            cuda.syncthreads() 
        if t == 0 and q_a > q_max:
            q_max = q_a
            a_max_index = a_index
    if t == 0 and a_so_far != a_max_index:
        policy[s_index] = a_max_index
        cuda.atomic.min(policy_stable, 0, int8(0))
        
@cuda.jit(void(float32[:,:,:,:], float32[:], float32[:], int16[:], float32, float32, float32[:]))
def jcr_pi_contraction_cuda_atomicmax_eval(P, V_in, V_out, policy, reward_car_moved, gamma, d):
    const_states = cuda.const.array_like(STATES)
    const_actions = cuda.const.array_like(ACTIONS)
    const_rewards = cuda.const.array_like(REWARDS)
    shared_v_new = cuda.shared.array(512, dtype=float32) # corresponds to DEFAULT_TPB
    shared_v_in = cuda.shared.array(2048, dtype=float32) # corresponds to MAX_N_STATES
    s_index = cuda.blockIdx.x
    t = cuda.threadIdx.x
    tpb = cuda.blockDim.x
    n_states = const_states.shape[0]
    n_rewards = const_rewards.shape[0]    
    nspt = (n_states + tpb - 1) // tpb # next states per thread
    next_state_index = t
    for _ in range(nspt):
        if next_state_index < n_states:
            shared_v_in[next_state_index] = V_in[next_state_index]
        next_state_index += tpb 
    cuda.syncthreads()
    a_index = policy[s_index]
    reward_cars_moved = math.fabs(const_actions[a_index]) * reward_car_moved
    v_new = float32(0.0)    
    for r_index in range(n_rewards):
        shared_v_new[t] = float32(0.0)
        r = reward_cars_moved + const_rewards[r_index]        
        next_state_index = t
        for _ in range(nspt):
            if next_state_index < n_states:
                shared_v_new[t] += P[r_index, next_state_index, s_index, a_index] * (r + gamma * shared_v_in[next_state_index]) 
            cuda.syncthreads()
            next_state_index += tpb                        
        stride = tpb >> 1
        while stride > 0: # sum-reduction
            if t < stride:
                shared_v_new[t] += shared_v_new[t + stride]
            cuda.syncthreads()
            stride >>= 1
        v_new += shared_v_new[0]
        cuda.syncthreads() 
    if t == 0:
        v = V_in[s_index]
        cuda.atomic.max(d, 0, math.fabs(v - v_new))
        V_out[s_index] = v_new           
        
@cuda.jit(void(float32[:,:,:,:], float32[:], int16[:], float32, float32, int32[:]))
def jcr_pi_contraction_cuda_atomicmax_improve(P, V, policy, reward_car_moved, gamma, policy_stable):
    const_states = cuda.const.array_like(STATES)
    const_actions = cuda.const.array_like(ACTIONS)
    const_rewards = cuda.const.array_like(REWARDS)
    shared_q_new = cuda.shared.array(512, dtype=float32) # corresponds to DEFAULT_TPB
    shared_v_in = cuda.shared.array(4096, dtype=float32) # corresponds to MAX_N_STATES
    s_index = cuda.blockIdx.x
    t = cuda.threadIdx.x
    tpb = cuda.blockDim.x
    n_states = const_states.shape[0]
    n_rewards = const_rewards.shape[0]    
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
        q_a = float32(0.0)
        reward_cars_moved = math.fabs(const_actions[a_index]) * reward_car_moved
        for r_index in range(n_rewards):
            shared_q_new[t] = float32(0.0)
            r = reward_cars_moved + const_rewards[r_index]        
            next_state_index = t
            for _ in range(nspt):
                if next_state_index < n_states:            
                    shared_q_new[t] += P[r_index, next_state_index, s_index, a_index] * (r + gamma * shared_v_in[next_state_index]) 
                cuda.syncthreads()
                next_state_index += tpb                        
            stride = tpb >> 1
            while stride > 0: # sum-reduction
                if t < stride:
                    shared_q_new[t] += shared_q_new[t + stride]
                cuda.syncthreads()
                stride >>= 1
            q_a += shared_q_new[0]
            cuda.syncthreads() 
        if t == 0 and q_a > q_max:
            q_max = q_a
            a_max_index = a_index
    if t == 0 and a_so_far != a_max_index:
        policy[s_index] = a_max_index
        cuda.atomic.min(policy_stable, 0, int8(0))        