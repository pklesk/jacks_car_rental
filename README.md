# CUDA implementations of policy iteration algorithm for Jack's Car Rental (JCR)
The repository constitutes a part of research on CUDA computational approaches for algorithms based on 
*contraction mapping theorem* due to Banach (a.k.a. the fixed-point theorem).

JCR, devised by Sutton and Barto (2020), is a classic problem of finding the optimal policy for 
a fully known MDP (Markov Decision Process), i.e., given its joint-probability distribution.
In this research, the *policy iteration* method has been applied, and the repository contains
four CUDA implementations of policy iteration (and two referential CPU-based implementation) for the JCR problem.

| parameter | description |
|:--------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------|
| <img src="extras/jcr_20_5_step_0.png"/> | <img src="extras/jcr_20_5_step_1.png"/> |                                                                            


## Acknowledgments and credits
- [Numba](https://numba.pydata.org): a high-performance just-in-time Python compiler.
