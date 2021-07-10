import numpy as np
from algorithms import LinUCB, RegretBalancingElim
from utils import FiniteContLinearRep, ContBanditProblem, make_random_rep, make_newlinrep_reshaped
import matplotlib.pyplot as plt
import time


nc, na, nd = 10, 5, 4
noise_std = 0.3
delta = 0.01
reg_val = 1
T = 10000

scale = 1
adaptive_ci = True




# question 1
# mutiple runs Linear UCB
runs = 15

runregs = np.zeros((runs, T))

linrep = make_random_rep(nc, na, nd, False)


for r in range(runs):
    print('run %s'%r)   
    linrep = make_random_rep(nc, na, nd, False)
    algo = LinUCB(linrep, reg_val, noise_std, delta)
    problem = ContBanditProblem(linrep, algo, noise_std)
    start_time = time.time()
    problem.run(T)
    print(f"--- finished in {np.round(time.time() - start_time,2)} seconds ---")
    reg = problem.exp_instant_regret.cumsum()   
    runregs[r] = reg
 
Regmean = runregs.mean(axis = 0)
plt.plot(Regmean, label="LinUCB")
plt.title('LinUCB simple_main regret over %s runs'%runs)





# mutliple runs RegBalAlg



runregsreg = np.zeros((runs, T))
for r in range(runs):
    print('run %s'%r)   
    linrep = make_random_rep(nc, na, nd, False)
    # reps_nested = [make_newlinrep_reshaped(linrep, i) for i in [5, 10, 25]]
    reps_random = [ make_random_rep(nc, na, i, False) for i in [nd+1] ]
    reps = reps_random + [linrep]
    algo = RegretBalancingElim(reps, reg_val, noise_std, delta)
    problem = ContBanditProblem(linrep, algo, noise_std)
    start_time = time.time()
    problem.run(T)
    print(f"--- finished in {np.round(time.time() - start_time,2)} seconds ---")
    reg = problem.exp_instant_regret.cumsum()
    runregsreg[r] = reg
 
Regmeanreg = runregsreg.mean(axis = 0)
plt.plot(Regmeanreg, label="RegBalElim")
plt.title('RegBalAlg simple_main regret over %s runs'%runs)


# rondom LiNUCB
runregsrandom = np.zeros((runs, T))
for r in range(runs):
    print('run %s'%r)   
    linrep = make_random_rep(nc, na, nd, False)
    
    reps_random = [ make_random_rep(nc, na, i, False) for i in [nd+1] ]
    # linUCB on random REP
    algo = LinUCB(reps_random[0], reg_val, noise_std, delta)
    problem = ContBanditProblem(linrep, algo, noise_std)
    start_time = time.time()
    problem.run(T)
    print(f"--- finished in {np.round(time.time() - start_time,2)} seconds ---")
    reg = problem.exp_instant_regret.cumsum()
    runregsrandom[r] = reg
 
Regmeanrandom = runregsrandom.mean(axis = 0)
plt.plot(Regmeanrandom, label="LinUCB-random")
plt.title('LinUCB-random simple_main regret over %s runs'%runs)




# comparison 
plt.plot(Regmean, label="LinUCB")
plt.plot(Regmeanreg, label="RegBalElim")
plt.plot(Regmeanrandom, label="LinUCB-random")
plt.title('simple_main regret over %s runs'%runs)
plt.legend()
plt.show()

