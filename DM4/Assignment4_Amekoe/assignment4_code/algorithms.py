import numpy as np 

import math
from math import log, sqrt

class LinUCB:

    def __init__(self, 
        representation,
        reg_val, noise_std, delta=0.01
    ):
        self.representation = representation
        self.reg_val = reg_val
        self.noise_std = noise_std
        self.param_bound = representation.param_bound
        self.features_bound = representation.features_bound
        self.delta = delta
        self.reset()

    def reset(self):
        ### TODO: initialize necessary info
        nc, na, nd = self.representation.features.shape
        self.inv_A_t = (1/self.reg_val)*np.eye(nd) # (1/lambda)*I_d ==> A_0 = lambda*I_d  and A = Sigma
        self.b_t = np.zeros(nd) # 0_d 
        
        
        # initialisation of alpha_t or beta_t
        self.alpha_t = np.sqrt(2*(self.noise_std**2)*np.log(1/self.delta)
                               )+ np.sqrt(self.reg_val)*self.param_bound
          
        
        self.param =  self.inv_A_t.dot(self.b_t) # theta_t
        ###################################
        self.t = 1

    def sample_action(self, context):
        ### TODO: implement action selection strategy
        
        nc, na, nd = self.representation.features.shape
        
        B_t = np.zeros(na) # the bonus 
        for i in range(na):
            phi_u_t = self.representation.features[context, i]
            B_t[i] = self.representation.features[context, i]@self.param + self.alpha_t*np.sqrt( 
                phi_u_t.dot(self.inv_A_t.dot(phi_u_t)) )
        maxa = np.argmax(B_t) 
        
        ###################################
        self.t += 1
        return maxa

    def update(self, context, action, reward):
        v = self.representation.get_features(context, action)
        ### TODO: update internal info (return nothing)
        
        # update of A_inv using Sherman-Morrison inversion Lemma
        numerator = self.inv_A_t.dot( np.outer(v,v).dot(self.inv_A_t) )
        denominator = 1+ v@ self.inv_A_t.dot(v)
        self.inv_A_t = self.inv_A_t - (1/denominator)*numerator
        
        
        self.b_t = self.b_t + reward*v # update of b_t 
        
        self.param =  self.inv_A_t.dot(self.b_t) #  update of theta_t
        
        # update of alpha_t or beta_t 
        self.alpha_t = np.sqrt(  2*(self.noise_std**2)*np.log( 
            1/( np.sqrt(np.linalg.det(self.inv_A_t) *self.reg_val)*self.delta )
            )  ) + np.sqrt(self.reg_val)*self.param_bound  
        
    
        
        ###################################


class RegretBalancingElim:
    def __init__(self, 
        representations,
        reg_val, noise_std,delta=0.01
    ):
        self.representations = representations
        self.reg_val = reg_val
        self.noise_std = noise_std
        self.param_bound = [r.param_bound for r in representations]
        self.features_bound = [r.features_bound for r in representations]
        self.delta = delta
        self.last_selected_rep = None
        self.active_reps = None # list of active (non-eliminated) representations
        self.t = None
        self.reset()
    

    def reset(self):
        ### TODO: initialize necessary info
        self.M = len(self.param_bound)
        self.active_reps = [i+1 for i in range(self.M)] # the active set representations: list  (of int ) 
        self.n =  np.zeros(self.M) # the number of times each learner was played 
        self.U =  np.zeros(self.M)  # the reward of each learner
        self.R =  np.zeros(self.M) # the pseudo regret
        self.ind = 0 # different to index in the representations table 
        
        # initialisation of learners 
        self.learners = [LinUCB(linrep,self.reg_val, self.noise_std, self.delta)
                         for linrep in  self.representations
                         ]
        
        ###################################
        self.t = 1
    
    def optimistic_action(self, rep_idx, context):
        ### TODO: implement action selection strategy given the selected representation
        
        # we use the LinUCB class 
        maxa = self.learners[rep_idx].sample_action(context)

        ###################################
        return maxa

    def sample_action(self, context):
        ### TODO: implement representation selection strategy
        #         and action selection strategy
        
        self.ind = np.argmin(self.R) # the index of the representation with the min lower bound
        
        # the ind of selected in the representations. -1 because  we added +1 above
        self.last_selected_rep = self.active_reps[self.ind]-1
        action = self.optimistic_action(self.last_selected_rep, context)
        
        
        ###################################
        self.t += 1
        return action

    def update(self, context, action, reward):
        idx = self.last_selected_rep
        v = self.representations[idx].get_features(context, action)
        ### TODO: implement update of internal info and active set

        self.n[self.ind] = self.n[self.ind]+1 # n_i(t) = n_i(t-1)+1
        self.U[self.ind] =  self.U[self.ind] + reward # U_i(t) = U_i(t-1) + r
        
        # update the the selected learner and R_i(n_i(t))
        self.learners[idx].update(context, action, reward)
        alpha_t = self.learners[idx].alpha_t
        A_inv_t = self.learners[idx].inv_A_t
        self.R[self.ind] = self.R[self.ind] + 2*alpha_t*np.sqrt( v.T.dot(A_inv_t.dot(v)))
        
        
        # elimination and update of the active set 
        c = 1 #!!!
        
        lower_bounds = np.zeros(len(self.active_reps))
        # we suppose every learner was played at least one time before the elimination 
        # to avoid somme nummerical issue
        # to avoid numerical issue
        if (self.n>0).all(): 
            for ind_j, j in enumerate(self.active_reps):
                
                # the lower bounds give of learnears at the round t
                lower_bounds[ind_j] = (self.U[ind_j]/self.n[ind_j])- c*np.sqrt(
                    np.log( self.M*np.log(self.n[ind_j])/self.delta ) / self.n[ind_j])
                
                # max lower bound 
            lower_bound_max  = np.max(lower_bounds)
            
            for ind_i, i in enumerate(self.active_reps):
                
                upper_bound_i = (self.U[ind_i]/self.n[ind_i])+ (self.R[ind_i]/self.n[ind_i]) +c*np.sqrt(
                    np.log( self.M*np.log(self.n[ind_i])/self.delta )/self.n[ind_i] )
                if upper_bound_i < lower_bound_max:
                    remove_learner = self.active_reps.pop(ind_i) 
                    #self.active_reps.remove(i)
                    self.n = np.delete(self.n,ind_i)# the nomber of time the eliminated learner was played
                    self.R = np.delete(self.R,ind_i)
                    self.U = np.delete(self.U, ind_i)
                    
        ###################################
