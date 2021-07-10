import numpy as np
from algorithms import LinUCB, RegretBalancingElim
from utils import FiniteContLinearRep, ContBanditProblem, make_random_rep, make_newlinrep_reshaped
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed

def execute_regbalelim(T, true_rep, reps, reg_val, noise_std, delta, num):
    """
    We modifie the function a bit to return of learners that remains in the active set
    """
    algo = RegretBalancingElim(reps, reg_val, noise_std, delta)
    problem = ContBanditProblem(true_rep, algo, noise_std)
    start_time = time.time()
    problem.run(T)
    print(f"--- {num} finished in {np.round(time.time() - start_time,2)} seconds ---")
    reg = problem.exp_instant_regret.cumsum()
    return reg, algo.active_reps

def execute_linucb(T, true_rep, rep, reg_val, noise_std, delta, num):
    algo = LinUCB(rep, reg_val, noise_std, delta)
    problem = ContBanditProblem(true_rep, algo, noise_std)
    start_time = time.time()
    problem.run(T)
    print(f"--- {num} finished in {np.round(time.time() - start_time,2)} seconds ---")
    reg = problem.exp_instant_regret.cumsum()
    return reg

PARALLEL = True
NUM_CORES = 6
NRUNS = 5
nc, na, nd = 200, 20, 20
noise_std = 0.3
reg_val = 1
delta = 0.01
T = 50000

b = np.load('final_representation2.npz')
reps = []
for i in range(b['true_rep']+1):
    feat = b[f'reps_{i}']
    param = b[f'reps_param_{i}']
    reps.append(FiniteContLinearRep(feat, param))
linrep = reps[b['true_rep']]

print("Running algorithm RegretBalancingElim")
results_reg_active  = []
if PARALLEL:
    results_reg_active = Parallel(n_jobs=NUM_CORES)(
        delayed(execute_regbalelim)(T, linrep, reps, reg_val, noise_std, delta, i) for i in range(NRUNS)
    )
else:
    for n in range(NRUNS):
        results_reg_active.append(
            execute_regbalelim(T, linrep, reps, reg_val, noise_std, delta, n)
        )

active_last_learns  = [results_reg_active[i][1] for i in range(NRUNS)] # list of active learners at each run
resultsr = [results_reg_active[i][0] for i in range(NRUNS)] # regrets : list of array


# question 3-c
print(active_last_learns)


regrets = []
for el in resultsr:
    regrets.append(el.tolist())
regrets = np.array(regrets)
mean_regretreg = regrets.mean(axis=0)
std_regretreg = regrets.std(axis=0) / np.sqrt(regrets.shape[0])
plt.plot(mean_regretreg, label="RegBalElim")
plt.fill_between(np.arange(T), mean_regretreg - 2*std_regretreg, mean_regretreg + 2*std_regretreg, alpha=0.1)




print("Running algorithm LinUCB")

result_ucbs = []

for nf, f in enumerate(reps):

    results = []
    if PARALLEL:
        results = Parallel(n_jobs=NUM_CORES)(
            delayed(execute_linucb)(T, linrep, f, reg_val, noise_std, delta, i) for i in range(NRUNS)
        )
    else:
        for n in range(NRUNS):
            results.append(
                execute_linucb(T, linrep, f, reg_val, noise_std, delta, n)
            )
    result_ucbs.append(results)




# question 3-a 
for nf, results in enumerate(result_ucbs):
    regrets = []
    for el in results:
        regrets.append(el.tolist())
    regrets = np.array(regrets)
    mean_regret = regrets.mean(axis=0)
    std_regret = regrets.std(axis=0) / np.sqrt(regrets.shape[0])
    plt.plot(mean_regret, label=f"LinUCB - f{nf+1}")
    plt.fill_between(np.arange(T), mean_regret - 2*std_regret, mean_regret + 2*std_regret, alpha=0.1)
    # after ploting the all the learnes with LinUCB algo, we plot the one of RegBalAlg
    if nf==len(reps)-1 :
        plt.plot(mean_regretreg, label="RegBalElim")
        plt.fill_between(np.arange(T), mean_regretreg - 2*std_regretreg, mean_regretreg + 2*std_regretreg, alpha=0.1)
plt.legend()
plt.show()



# question 3-d

regrets = []
for el in result_ucbs[-1]:
    
    regrets.append(el.tolist())
regrets = np.array(regrets)
mean_regret = regrets.mean(axis=0)
std_regret = regrets.std(axis=0) / np.sqrt(regrets.shape[0])
plt.plot(mean_regret, label=f"LinUCB - f{7+1}")
plt.fill_between(np.arange(T), mean_regret - 2*std_regret, mean_regret + 2*std_regret, alpha=0.1)
    # after ploting the all the learnes with LinUCB algo, we plot the one of RegBalAlg

plt.plot(mean_regretreg, label="RegBalElim")
plt.fill_between(np.arange(T), mean_regretreg - 2*std_regretreg, mean_regretreg + 2*std_regretreg, alpha=0.1)
plt.legend()
plt.show()


# optional plot the two best learners f3- and f8
for nf, results in enumerate(result_ucbs):
    if nf ==2 or nf==7:
        regrets = []
        for el in results:
            regrets.append(el.tolist())
        regrets = np.array(regrets)
        mean_regret = regrets.mean(axis=0)
        std_regret = regrets.std(axis=0) / np.sqrt(regrets.shape[0])
        plt.plot(mean_regret, label=f"LinUCB - f{nf+1}")
        plt.fill_between(np.arange(T), mean_regret - 2*std_regret, mean_regret + 2*std_regret, alpha=0.1)
        # after ploting the all the learnes with LinUCB algo, we plot the one of RegBalAlg
plt.legend()
plt.show()
