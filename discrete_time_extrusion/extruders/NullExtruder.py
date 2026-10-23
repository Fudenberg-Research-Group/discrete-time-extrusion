class NullExtruder():
    
    def __init__(self,
                 number,
                 barrier_engine,
                 chromosome_bounds,
                 *args, **kwargs):
            
        self.number = number
        self.barrier_engine = barrier_engine
        
        self.xp = barrier_engine.xp
        self.get_list = barrier_engine.get_list
        
        self.lattice_size = barrier_engine.lattice_size
        self.sites = self.xp.arange(self.lattice_size, dtype=self.xp.int32)
        self.occupied = self.xp.zeros(self.lattice_size, dtype=self.xp.uint32)

        self.states = self.xp.zeros(self.number, dtype=self.xp.int32)
        self.directions = self.xp.zeros(self.number, dtype=self.xp.uint32)
        
        self.stalled = self.xp.zeros((self.number, 2), dtype=self.xp.uint32)
        self.positions = self.xp.zeros((self.number, 2), dtype=self.xp.int32) - 1
        
        self.chromosome_bounds = self.xp.asarray(chromosome_bounds, dtype=self.xp.int32)

        self.update_occupancies()
        

    def step(self, *args, **kwargs):
    
        self.barrier_engine.step(self)
        
    
    def steps(self, N, *args, **kwargs):
    
        for _ in range(N):
            self.step(*args, **kwargs)
        

    def update_occupancies(self):
        
        self.occupied.fill(0)
        self.occupied[self.chromosome_bounds] = 1

        ids = self.positions[self.xp.greater_equal(self.positions, 0)]
        self.occupied[ids] = 1
        

    def get_states(self):

        return self.get_list(self.states)
        
        
    def get_positions(self):

        return self.get_list(self.positions)
        
        
    def get_bound_positions(self):

        ids = self.xp.greater_equal(self.positions, 0).all(axis=1)
        bound_positions = self.positions[ids]

        return self.get_list(bound_positions)
