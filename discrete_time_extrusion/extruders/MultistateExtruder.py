from . import BaseExtruder
    

class MultistateExtruder(BaseExtruder.BaseExtruder):
    
    def __init__(self,
                 number,
                 barrier_engine,
                 chromosome_bounds,
                 birth_prob,
                 death_prob,
                 stalled_death_prob,
                 diffusion_prob,
                 step_prob,
                 pause_prob,
                 *args, **kwargs):
    
        super().__init__(number,
                         barrier_engine,
                         chromosome_bounds,
                         birth_prob,
                         death_prob,
                         stalled_death_prob,
                         diffusion_prob,
                         step_prob,
                         pause_prob,
                         *args, **kwargs)
        
        self.state_dict = kwargs["LEF_states"]
        self.transition_dict = kwargs["LEF_transitions"]
        

    def state_transitions(self, unbound_state_id, n_max=3, **kwargs):
        
        ids_array = self.xp.zeros((len(self.state_dict), self.number), dtype=self.xp.int32)
        products_array = self.xp.zeros((len(self.state_dict), self.number), dtype=self.xp.int32)
            
        for i, state_id in enumerate(self.state_dict.values()):
            ctr = 0
            
            state_array = self.xp.zeros(n_max+2, dtype=self.xp.int32) - 1
            transition_array = self.xp.zeros((n_max+1, self.number), dtype=self.xp.float32)
            
            for ids, transition_prob in self.transition_dict.items():
                if state_id == int(ids[0]):
                    state_array[ctr] = int(ids[1])
                    transition_array[ctr] = transition_prob[self.positions].max(axis=1)
                    
                    ctr += 1
                    
                    if ctr > n_max:
                        raise RuntimeError("State %d found with more than %d candidate reactions - set higher n_max" % (state_id, n_max))
                    
            if state_id == max(self.state_dict.values()):
                death_prob = self.xp.where(self.stalled,
                                           self.stalled_death_prob[self.positions],
                                           self.death_prob[self.positions])
                                      
                state_array[ctr] = unbound_state_id
                transition_array[ctr] = death_prob.max(axis=1)

            # Vectorized discrete-time Gillespie algorithm
            rng = self.xp.random.random(self.number)
            
            cumul_prob = self.xp.cumsum(transition_array, axis=0)
            product_ids = self.xp.less(cumul_prob, rng).sum(axis=0)
            
            if self.xp.greater(cumul_prob, 1).any():
                raise RuntimeError("Found overflowing rates - set smaller lattice timestep")
            
            product_states = state_array[product_ids]
            transition_mask = self.xp.less(product_ids, n_max) * self.xp.equal(self.states, state_id)
        
            ids = self.xp.flatnonzero(transition_mask)
            products = product_states[ids]
            
            ids_array[i] = self.xp.r_[ids, self.xp.full(self.number-len(ids), -1, dtype=self.xp.int32)]
            products_array[i] = self.xp.r_[products, self.xp.full(self.number-len(ids), -1, dtype=self.xp.int32)]
            
        return ids_array, products_array
            
        
    def chemical_step(self, unbound_state_id=0, bound_state_id=1, **kwargs):

        ids_array, products_array = self.state_transitions(unbound_state_id, **kwargs)
        
        ids_birth = self.birth(unbound_state_id)
        self.states[ids_birth] = bound_state_id
	
        rng = self.xp.less(self.xp.random.random(self.number), 0.5).astype(self.xp.uint32)

        for ids, products in zip(ids_array, products_array):
            is_product = self.xp.greater_equal(ids, 0)
            is_active = self.xp.equal(products, self.state_dict['RN'])
                        
            self.states[ids[is_product]] = products[is_product]
            self.directions[ids] = self.xp.where(is_product*is_active, rng, self.directions[ids])
            
        ids_death = ids[self.xp.equal(products, unbound_state_id)]
        
        self.unload(ids_death)
        self.update_occupancies()


    def step(self, **kwargs):
    
        super().step(active_state_id=self.state_dict['RN'], **kwargs)
