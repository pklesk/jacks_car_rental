Large binary (pickled) files with with probability distribution 
for an MDP describing the "Jack's Car Rental" problem 
can be generated into this folder via:

P = jcr.mdp_joint_distr_jcr()
pickle_objects(jcr.MDP_PATH, [P])