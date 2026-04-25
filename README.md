# CUDA implementations of policy iteration algorithm for Jack's Car Rental (JCR)
The repository constitutes a part of research on CUDA computational approaches for algorithms based on 
*contraction mapping theorem* due to Banach (a.k.a. the fixed-point theorem).

JCR, devised by Sutton and Barto (2020), is a classic problem of finding the optimal policy for 
a fully known MDP (Markov Decision Process), i.e., given its joint-probability distribution.
In this research, the *policy iteration* method has been applied, and the repository contains
four CUDA implementations of policy iteration (and two referential CPU-based implementation) for the JCR problem.

<table>
  <tr>
    <td>$V_0, \pi_0\quad$<br/><br/><img src="extras/jcr_20_5_step_0.png"/></td>
    <td>$\xrightarrow{\textrm{E}}V_1\approx v_{\pi_0}\xrightarrow{\textrm{I}}\pi_1\quad$<br/><img src="extras/jcr_20_5_step_1.png"/></td>
    <td>$\cdots$<br/></td>
    <td>$\xrightarrow{\textrm{E}}V_4\approx v_{\pi_3}\xrightarrow{\textrm{I}}\pi_4=\pi^*\quad$<br/><img src="extras/jcr_20_5_step_4.png"/></td>
  </tr>
</table>

- [Numba](https://numba.pydata.org): a high-performance just-in-time Python compiler.

END
