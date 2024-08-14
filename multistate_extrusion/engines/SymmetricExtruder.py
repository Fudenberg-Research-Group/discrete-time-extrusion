import numpy as np


class SymmetricExtruder():
    
    def __init__(self,
                 num_LEF,
                 barrier_engine,
                 birth_prob,
                 death_prob,
                 stalled_death_prob,
                 pause_prob,
                 *args, **kwargs):
    
        self.num_site = len(birth_prob)
        self.num_LEF = num_LEF
        
        self.birth_prob = birth_prob
        self.death_prob = death_prob

        self.pause = pause_prob
        self.stalled_death_prob = stalled_death_prob

        self.barrier = barrier_engine
        self.sites = np.arange(self.num_site, dtype=int)

        # LEF state equals 0 if LEF is unbound, 1 if bound
        self.states = np.zeros(self.num_LEF, dtype=int)
        self.positions = np.zeros((self.num_LEF, 2), dtype=int) - 1
        
        self.occupied = np.zeros(self.num_site, dtype=bool)
        self.stalled = np.zeros((self.num_LEF, 2), dtype=bool)

        self.occupied[0] = self.occupied[-1] = True
        

    def lef_step(self, active_state_id):
    
        for i in range(self.num_LEF):
            if self.states[i] == active_state_id:
                stall1 = self.barrier.prob_left[self.positions[i, 0]]
                stall2 = self.barrier.prob_right[self.positions[i, 1]]
                                        
                if np.random.random() < stall1:
                    self.stalled[i, 0] = True
                if np.random.random() < stall2:
                    self.stalled[i, 1] = True
                             
                cur1, cur2 = self.positions[i]
                
                if not self.stalled[i, 0]:
                    if not self.occupied[cur1-1]:
                        pause1 = self.pause[cur1]
                        
                        if np.random.random() > pause1:
                            self.occupied[cur1 - 1] = True
                            self.occupied[cur1] = False
                            
                            self.positions[i, 0] = cur1 - 1
                            
                if not self.stalled[i, 1]:
                    if not self.occupied[cur2 + 1]:
                        pause2 = self.pause[cur2]
                        
                        if np.random.random() > pause2:
                            self.occupied[cur2 + 1] = True
                            self.occupied[cur2] = False
                            
                            self.positions[i, 1] = cur2 + 1
                            
                            
    def lef_birth(self, unbound_state_id):
    
        free_sites = self.sites[~self.occupied]
        binding_sites = np.random.choice(free_sites, size=self.num_LEF, replace=False)

        rng = np.random.random(self.num_LEF) < self.birth_prob[binding_sites]
        ids = np.flatnonzero(rng * (self.states == unbound_state_id))
                
        if len(ids) > 0:
            self.occupied[binding_sites[ids]] = True
            self.positions[ids] = binding_sites[ids, None]
        
            rng_stagger = (np.random.random(len(ids)) < 0.5) * ~self.occupied[binding_sites[ids]+1]

            self.positions[ids, 1] = np.where(rng_stagger,
                                                  self.positions[ids, 1] + 1,
                                                  self.positions[ids, 1])
                                                  
            self.occupied[binding_sites[ids]+1] = np.where(rng_stagger,
                                                           True,
                                                           self.occupied[binding_sites[ids]+1])
                                                           
        return ids
                                                                                
        
    def lef_death(self, bound_state_id):
    
        death_prob = np.where(self.stalled,
                              self.stalled_death_prob[self.positions],
                              self.death_prob[self.positions])
                              
        death_prob = np.max(death_prob, axis=1)
        
        rng = np.random.random(self.num_LEF) < death_prob
        ids = np.flatnonzero(rng * (self.states == bound_state_id))
        
        return ids
        

    def update_LEF_arrays(self, ids_death):
    
        self.stalled[ids_death] = False
        self.occupied[self.positions[ids_death]] = False
        
        self.positions[ids_death] = -1
        
        
    def update_states(self, unbound_state_id, bound_state_id):
    
        ids_birth = self.lef_birth(unbound_state_id)
        ids_death = self.lef_death(bound_state_id)
        
        self.states[ids_birth] = bound_state_id
        self.states[ids_death] = unbound_state_id

        self.update_LEF_arrays(ids_death)


    def step(self, unbound_state_id=0, bound_state_id=1, active_state_id=1):
    
        self.barrier.step(self)

        self.update_states(unbound_state_id, bound_state_id)
        self.lef_step(active_state_id)
        
    
    def steps(self, N):
    
        for _ in range(N):
            self.step()
