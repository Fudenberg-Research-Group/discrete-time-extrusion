from . import NullExtruder, EngineFactory
    

class BaseExtruder(NullExtruder.NullExtruder):
    
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
    
        super().__init__(number, barrier_engine, chromosome_bounds, *args, **kwargs)
		
        self.birth_prob = birth_prob
        
        self.death_prob = death_prob
        self.stalled_death_prob = stalled_death_prob
        
        self.diffusion_prob = diffusion_prob
        self.step_prob = step_prob
        self.pause_prob = pause_prob

        self.stepping_engine = EngineFactory.SteppingEngine
        self.diffusion_engine = EngineFactory.DiffusionEngine
        
                            
    def birth(self, unbound_state_id):
    
        unbound_ids = self.xp.flatnonzero(self.xp.equal(self.states, unbound_state_id))
        
        free_sites = self.sites[self.xp.equal(self.occupied, 0) * self.xp.greater(self.birth_prob, 0)]
        binding_sites = self.xp.random.choice(free_sites, size=len(unbound_ids), replace=False)

        rng = self.xp.less(self.xp.random.random(len(unbound_ids)), self.birth_prob[binding_sites])
        
        load_ids = self.xp.flatnonzero(rng)
        ids = unbound_ids[load_ids]

        if len(load_ids) > 0:
            binding_sites = binding_sites[load_ids]
            available_mask = ~self.occupied[binding_sites+1] * ~self.xp.in1d(binding_sites+1, binding_sites)
			
            rng_dir = self.xp.less(self.xp.random.random(len(ids)), 0.5)
            rng_stagger = self.xp.less(self.xp.random.random(len(ids)), 0.5) * available_mask

            self.positions[ids] = binding_sites[:, None]
            self.positions[ids, 1] = self.xp.where(rng_stagger,
                                                   self.positions[ids, 1] + 1,
                                                   self.positions[ids, 1])
                                                   
            self.directions[ids] = rng_dir.astype(self.xp.uint32)
                                                           
        return ids
                                                                                
        
    def death(self, bound_state_id):
    
        death_prob = self.xp.where(self.stalled,
                                   self.stalled_death_prob[self.positions],
                                   self.death_prob[self.positions])
        death_prob = self.xp.max(death_prob, axis=1)
        
        rng = self.xp.less(self.xp.random.random(self.number), death_prob)
        ids = self.xp.flatnonzero(rng * self.xp.equal(self.states, bound_state_id))
        
        return ids
        
	
    def unload(self, ids_death):
    
        self.stalled[ids_death] = False
        self.positions[ids_death] = -1
        
        
    def chemical_step(self, unbound_state_id=0, bound_state_id=1, **kwargs):
    
        ids_birth = self.birth(unbound_state_id)
        ids_death = self.death(bound_state_id)
        
        self.states[ids_birth] = bound_state_id
        self.states[ids_death] = unbound_state_id

        self.unload(ids_death)
        self.update_occupancies()


    def step(self, tension=0, unbound_state_id=0, active_state_id=1, **kwargs):

        super().step(**kwargs)
        
        self.chemical_step(**kwargs)
                
        if isinstance(tension, float) | isinstance(tension, int):
            tension_array = self.xp.full((self.number, 2), tension, dtype=self.xp.float64)
        elif isinstance(tension, self.xp.ndarray):
            tension_array = self.xp.asarray(tension, dtype=self.xp.float64)

            if tension_array.shape == (self.number,):
                tension_array = self.xp.tile(tension_array[:, None], (1, 2))
            elif tension_array.shape != (self.number, 2):
                raise RuntimeError("Misshapen tension array - expected %d or %dx2" % (self.number, self.number))

        self.stepping_engine(self, tension_array, active_state_id, **kwargs)
        self.diffusion_engine(self, tension_array, unbound_state_id, **kwargs)
