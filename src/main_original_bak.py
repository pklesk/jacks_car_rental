import itertools
import numpy as np
import time
import pickle
from matplotlib import pyplot as plt
from scipy.ndimage import label

MAX_CARS_AT_LOC = 20
MAX_CARS_MOVED = 5
REWARD_CAR_RENTED = 10.0
REWARD_CAR_MOVED = -2.0
GAMMA = 0.9
EPS = 1e-1
TOLERANCE_V = 1e-7
STATES = np.array(list(itertools.product(np.arange(0, MAX_CARS_AT_LOC + 1), np.arange(0, MAX_CARS_AT_LOC + 1))))
ACTIONS = np.arange(-MAX_CARS_MOVED, MAX_CARS_MOVED + 1)
REWARDS = np.arange(2 * MAX_CARS_AT_LOC + 1) * REWARD_CAR_RENTED

UNPICKLE_MDP = True
MDP_PATH = f"mdp_joint_distr_jcr_{MAX_CARS_AT_LOC}_{MAX_CARS_MOVED}.pkl"

LAMBDAS = [2.0, 3.0, 4.0]
MAX_N = MAX_CARS_AT_LOC
POISSON_DISTRS = {l: np.array([l**n * np.exp(-l) / np.math.factorial(n) for n in range(MAX_N + 1)]) for l in LAMBDAS}
for k in POISSON_DISTRS.keys():
    distr = POISSON_DISTRS[k]
    distr[-1] = 1.0 - np.sum(distr[:-1])
REQUEST_DISTRS_AT_LOCS = [POISSON_DISTRS[3.0], POISSON_DISTRS[4.0]]
RETURN_DISTRS_AT_LOCS = [POISSON_DISTRS[3.0], POISSON_DISTRS[2.0]]

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

def mdp_joint_distr_jcr():
    print("MDP_JOINT_DISTR_JCR...")
    t1 = time.time()
    P = np.zeros((REWARDS.shape[0], STATES.shape[0], STATES.shape[0], ACTIONS.shape[0]))
    print(f"[shape of P to prepare: {P.shape}, entries: {P.size}]")
    for s_index in range(STATES.shape[0]):
        print(f"PROGRESS: {s_index + 1}/{STATES.shape[0]}.")
        s_cars_at_0, s_cars_at_1 = STATES[s_index]
        for a_index in range(ACTIONS.shape[0]):
            a = ACTIONS[a_index]
            cars_moved = min(s_cars_at_0, a) if a > 0 else min(s_cars_at_1, -a)
            cars_at_0 = s_cars_at_0 - np.sign(a) * cars_moved
            cars_at_1 = s_cars_at_1 + np.sign(a) * cars_moved                
            cars_at_0 = min(max(0, cars_at_0), MAX_CARS_AT_LOC) # TODO max(0, ...) can be removed line 65 guarantees non-negative values 
            cars_at_1 = min(max(0, cars_at_1), MAX_CARS_AT_LOC) # TODO max(0, ...) can be removed line 65 guarantees non-negative values                                   
            for request_index in range(STATES.shape[0]):
                request = STATES[request_index]
                P_request = REQUEST_DISTRS_AT_LOCS[0][request[0]] * REQUEST_DISTRS_AT_LOCS[1][request[1]] 
                rented_at_0 = min(request[0], cars_at_0)
                rented_at_1 = min(request[1], cars_at_1)
                reward_index = rented_at_0 + rented_at_1                                       
                for return_index in range(STATES.shape[0]):
                    returned_at_0, returned_at_1 = STATES[return_index]
                    P_return = RETURN_DISTRS_AT_LOCS[0][returned_at_0] * RETURN_DISTRS_AT_LOCS[1][returned_at_1]                    
                    cars_at_0_next = min(cars_at_0 - rented_at_0 + returned_at_0, MAX_CARS_AT_LOC)
                    cars_at_1_next = min(cars_at_1 - rented_at_1 + returned_at_1, MAX_CARS_AT_LOC)
                    s_next_index = cars_at_0_next * (MAX_CARS_AT_LOC + 1) + cars_at_1_next                                                              
                    P[reward_index, s_next_index, s_index, a_index] += P_request * P_return                                                             
    t2 = time.time()
    print(f"MDP_JOINT_DISTR_JCR DONE. [time: {t2 - t1} s]")
    t1 = time.time()    
    return P
         
def policy_iteration_jcr(P, plots=False):
    print("POLICY_ITERATION_JCR...")
    t1 = time.time()    
    # INTIALIZATION
    V = np.zeros(STATES.shape[0])
    # policy = np.zeros(STATES.shape[0], dtype=np.int16) + MAX_CARS_MOVED # corresponds to action: move 0 cars (hence the shift)
    np.random.seed(0) 
    policy = np.random.randint(-MAX_CARS_MOVED, MAX_CARS_MOVED + 1, size=STATES.shape[0], dtype=np.int16) + MAX_CARS_MOVED
    if plots:
        print(ACTIONS[policy].reshape((MAX_CARS_AT_LOC + 1, MAX_CARS_AT_LOC + 1)))
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
            for s_index in range(STATES.shape[0]):
                v = V[s_index]
                a_index = policy[s_index]
                reward_cars_moved = np.abs(ACTIONS[a_index]) * REWARD_CAR_MOVED
                v_new = 0.0
                for r_index in range(REWARDS.shape[0]):
                    r = reward_cars_moved + REWARDS[r_index]
                    v_new += np.sum(P[r_index, :, s_index, a_index] * (r + GAMMA * V))
                V[s_index] = v_new
                d = max(d, np.abs(v - v_new))
            i_eval += 1
            t2_eval = time.time()
            print(f"POLICY EVALUATION ITERATION {i_eval} [d_inf: {d}, time: {t2_eval - t1_eval} s].")
            if d <= EPS:
                break
        # POLICY IMPROVEMENT
        t1_impr = time.time()
        policy_stable = True
        for s_index in range(STATES.shape[0]):
            a_so_far = policy[s_index]
            a_index = policy[s_index]
            qs = np.zeros(ACTIONS.shape[0])
            for a_index in range(ACTIONS.shape[0]):
                reward_cars_moved = np.abs(ACTIONS[a_index]) * REWARD_CAR_MOVED
                q = 0.0
                for r_index in range(REWARDS.shape[0]):
                    r = reward_cars_moved + REWARDS[r_index]
                    q += np.sum(P[r_index, :, s_index, a_index] * (r + GAMMA * V))
                qs[a_index] = q
            policy[s_index] = np.argmax(qs)
            if policy[s_index] != a_so_far:
                policy_stable = False 
        t2_impr = time.time()
        print(f"POLICY IMPROVEMENT [policy_stable: {policy_stable}, d_inf: {d}, d_inf < TOLERANCE_V: {d < TOLERANCE_V}, time: {t2_impr - t1_impr} s].")
        if plots:
            print(ACTIONS[policy].reshape((MAX_CARS_AT_LOC + 1, MAX_CARS_AT_LOC + 1)))
            plot_value_and_policy_jcr(V, policy)
        if policy_stable or (i_eval == 1 and d <= TOLERANCE_V):
            break
        i_main += 1
    t2 = time.time()    
    print(f"POLICY_ITERATION_JCR DONE. [time: {t2 - t1} s, main loop iterations: {i_main}]")
    return V, policy                                     

def plot_value_and_policy_jcr(V, policy):
    figsize = (12, 6)    
    fig = plt.figure(figsize=figsize)
   
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    V_to_plot = V.reshape((MAX_CARS_AT_LOC + 1, MAX_CARS_AT_LOC + 1))
    X, Y = np.meshgrid(np.arange(V_to_plot.shape[0]), np.arange(V_to_plot.shape[1]))
    ax1.plot_surface(X, Y, V_to_plot, cmap="viridis", edgecolor="k", linewidth=0.5)
    ax1.set_xlabel("CARS AT LOCATION 2")
    ax1.set_ylabel("CARS AT LOCATION 1")
    ax1.set_zlabel("VALUE")
    ax1.set_title("VALUE FUNCTION (V)")    
    ax2 = fig.add_subplot(1, 2, 2)
    policy_to_plot = ACTIONS[policy].reshape((MAX_CARS_AT_LOC + 1, MAX_CARS_AT_LOC + 1))    
    extent = [-0.5, MAX_CARS_AT_LOC + 0.5, -0.5, MAX_CARS_AT_LOC + 0.5]
    im = ax2.imshow(policy_to_plot, cmap="coolwarm", origin="lower", 
               vmin=-MAX_CARS_MOVED, vmax=MAX_CARS_MOVED, extent=extent)       
    for act in range(-MAX_CARS_MOVED, MAX_CARS_MOVED + 1):
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
    ax2.set_yticks(np.arange(0, MAX_CARS_AT_LOC + 1, 5)) 
    ax2.set_xticks(np.arange(0, MAX_CARS_AT_LOC + 1, 5))        
    ax2.set_title("POLICY")
    plt.subplots_adjust(wspace=0.5)              
    plt.show()
            
if __name__ == '__main__':
    print("JACK'S CAR RENTAL STARTING...")
    print(f"STATES' SHAPE: {STATES.shape}")
    if UNPICKLE_MDP:
        [P] = unpickle_objects(MDP_PATH)
    else:
        P = mdp_joint_distr_jcr()
        pickle_objects(MDP_PATH, [P])
    print(f"JOINT DISTRIBUTION -> SHAPE: {P.shape}, TYPE: {P.dtype}")    
    V, policy = policy_iteration_jcr(P, plots=False)
    plot_value_and_policy_jcr(V, policy)    
    print("JACK'S CAR RENTAL DONE.")