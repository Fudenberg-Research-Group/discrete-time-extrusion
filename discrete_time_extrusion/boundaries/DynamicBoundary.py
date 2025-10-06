from . import StaticBoundary


class DynamicBoundary(StaticBoundary.StaticBoundary):
     
    def __init__(self,
                 stall_left,
                 stall_right,
                 birth_prob,
                 death_prob,
                 *args, **kwargs):
        
        super().__init__(stall_left, stall_right)
        
        self.birth_prob = birth_prob
        self.death_prob = death_prob
            
        occupancy = birth_prob / (birth_prob + death_prob)

        self.states_left = self.xp.greater(stall_left, 0)
        self.states_right = self.xp.greater(stall_right, 0)
                
        rng_left = self.xp.less(self.xp.random.random(self.lattice_size), occupancy)
        rng_right = self.xp.less(self.xp.random.random(self.lattice_size), occupancy)
        
        self.states_left = self.xp.where(self.states_left,
                                         self.states_left*rng_left,
                                         -1)
        self.states_right = self.xp.where(self.states_right,
                                          self.states_right*rng_right,
                                          -1)

        self.stall_left = self.xp.equal(self.states_left, 1).astype(self.xp.float64)
        self.stall_right = self.xp.equal(self.states_right, 1).astype(self.xp.float64)
                         

    def birth(self, unbound_state_id):
    
        rng_left = self.xp.less(self.xp.random.random(self.lattice_size), self.birth_prob)
        rng_right = self.xp.less(self.xp.random.random(self.lattice_size), self.birth_prob)

        ids_left = self.xp.flatnonzero(rng_left * self.xp.equal(self.states_left, unbound_state_id))
        ids_right = self.xp.flatnonzero(rng_right * self.xp.equal(self.states_right, unbound_state_id))
        
        self.stall_left[ids_left] = 1
        self.stall_right[ids_right] = 1
        
        return ids_left, ids_right
                
        
    def death(self, bound_state_id):

        rng_left = self.xp.less(self.xp.random.random(self.lattice_size), self.death_prob)
        rng_right = self.xp.less(self.xp.random.random(self.lattice_size), self.death_prob)

        ids_left = self.xp.flatnonzero(rng_left * self.xp.equal(self.states_left, bound_state_id))
        ids_right = self.xp.flatnonzero(rng_right * self.xp.equal(self.states_right, bound_state_id))
        
        self.stall_left[ids_left] = 0
        self.stall_right[ids_right] = 0
        
        return ids_left, ids_right

    
    def step(self, extrusion_engine, unbound_state_id=0, bound_state_id=1):
    
        ids_birth_left, ids_birth_right = self.birth(unbound_state_id)
        ids_death_left, ids_death_right = self.death(bound_state_id)

        self.states_left[ids_birth_left] = bound_state_id
        self.states_left[ids_death_left] = unbound_state_id

        self.states_right[ids_birth_right] = bound_state_id
        self.states_right[ids_death_right] = unbound_state_id

        lef_ids_left = self.xp.in1d(extrusion_engine.positions[:, 0], ids_death_left)
        lef_ids_right = self.xp.in1d(extrusion_engine.positions[:, 1], ids_death_right)

        extrusion_engine.stalled[:, 0] = self.xp.where(lef_ids_left, 0, extrusion_engine.stalled[:, 0])
        extrusion_engine.stalled[:, 1] = self.xp.where(lef_ids_right, 0, extrusion_engine.stalled[:, 1])
