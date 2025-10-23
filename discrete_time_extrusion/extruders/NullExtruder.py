class NullExtruder():
    
    def __init__(self,
                 number,
                 barrier_engine,
                 chromosome_boundaries,
                 *args, **kwargs):
            
        self.number = number
        self.barrier_engine = barrier_engine
        
        self.xp = barrier_engine.xp
        self.get_list = barrier_engine.get_list
        
        self.lattice_size = barrier_engine.lattice_size
        self.occupied = self.xp.zeros(self.lattice_size, dtype=bool)
        self.sites = self.xp.arange(self.lattice_size, dtype=self.xp.int32)
        
        self.states = self.xp.zeros(self.number, dtype=self.xp.int32)
        self.directions = self.xp.zeros(self.number, dtype=self.xp.uint32)
        
        self.stalled = self.xp.zeros((self.number, 2), dtype=self.xp.uint32)
        self.positions = self.xp.zeros((self.number, 2), dtype=self.xp.int32) - 1
        
        self.chromosome_bounds = self.xp.asarray(chromosome_boundaries, dtype=self.xp.int32)

        self.update_occupancies()
        

    def step(self, *args, **kwargs):
    
        self.barrier_engine.step(self)
        
    
    def steps(self, N, *args, **kwargs):
    
        for _ in range(N):
            self.step(*args, **kwargs)
            
            
    def resolve_overlaps(self):
		
        leg_distance = self.xp.roll(self.positions, 1, axis=1) - self.positions
        is_loop = self.xp.not_equal(leg_distance, 0)
		
        ids = self.positions[is_loop]
        _, first, counts = self.xp.unique(ids, return_index=True, return_counts=True)
        
        first_dupl_ids = first[self.xp.greater(counts, 1)]
        ids[first_dupl_ids] = self.xp.where(self.xp.mod(first_dupl_ids, 2),
                                            ids[first_dupl_ids]-1,
                                            ids[first_dupl_ids]+1)
		
        self.positions[is_loop] = ids
        

    def update_occupancies(self):
        
        self.occupied.fill(False)
        self.occupied[self.chromosome_bounds] = True
        
        self.resolve_overlaps()
        
        ids = self.positions[self.xp.greater_equal(self.positions, 0)]
        self.occupied[ids] = True
        

    def get_states(self):

        return self.get_list(self.states)
        
        
    def get_positions(self):

        return self.get_list(self.positions)
        
        
    def get_bound_positions(self):

        ids = self.xp.greater_equal(self.positions, 0).all(axis=1)
        bound_positions = self.positions[ids]

        return self.get_list(bound_positions)
