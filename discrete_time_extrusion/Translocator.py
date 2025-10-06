from . import arrays


class Translocator():

    def __init__(self,
                 extrusion_engine,
                 barrier_engine,
                 type_list,
                 site_types,
                 ctcf_left_positions,
                 ctcf_right_positions,
                 chromosome_bounds=[0,-1],
                 device='CPU',
                 **kwargs):

        if device == 'CPU':
            import numpy as xp
            
        elif device == 'GPU':
            try:
                import cupy as xp
                use_cuda = xp.cuda.is_available()
		
                if not use_cuda:
                    raise ImportError("Could not load CUDA environment")
	
            except:
                raise
                
        else:
            raise RuntimeError("Unrecognized device %s - use either 'CPU' or 'GPU'" % device)
    
        sites_per_replica = kwargs['monomers_per_replica'] * kwargs['sites_per_monomer']
        number_of_LEFs = (kwargs['number_of_replica'] * kwargs['monomers_per_replica']) // kwargs['LEF_separation']
        
        assert len(site_types) == sites_per_replica, ("Site type array (%d) doesn't match replica lattice size (%d)"
                                                      % (len(site_types), sites_per_replica))

        self.time_unit = 1. / (kwargs['sites_per_monomer'] * kwargs['velocity_multiplier'])

        lef_arrays = arrays.make_LEF_arrays(xp, type_list, site_types, **kwargs)
        lef_transition_dict = arrays.make_LEF_transition_dict(xp, type_list, site_types, **kwargs)

        ctcf_dynamic_arrays = arrays.make_CTCF_dynamic_arrays(xp, type_list, site_types, **kwargs)
        ctcf_arrays = arrays.make_CTCF_arrays(xp, type_list, site_types,
                                              ctcf_left_positions, ctcf_right_positions, **kwargs)
                                              
        self.barrier_engine = barrier_engine(*ctcf_arrays, *ctcf_dynamic_arrays)
        self.extrusion_engine = extrusion_engine(number_of_LEFs, self.barrier_engine, chromosome_bounds,
												 *lef_arrays, **lef_transition_dict)
                
        kwargs['steps'] = int(kwargs['steps'] / self.time_unit)
        kwargs['dummy_steps'] = int(kwargs['dummy_steps'] / self.time_unit)
        
        self.params = kwargs
                

    def run(self, N, **kwargs):
            
        self.extrusion_engine.steps(N, self.params['mode'], **kwargs)
        
        
    def run_trajectory(self, period=None, steps=None, prune_unbound_LEFs=True, **kwargs):

        self.clear_trajectory()

        steps = int(steps) if steps else self.params['steps']
        period = int(period) if period else self.params['sites_per_monomer']
        
        self.run(self.params['dummy_steps']*period, **kwargs)
    
        for _ in range(steps):
            self.run(period, **kwargs)
            
            LEF_states = self.extrusion_engine.get_states()
            CTCF_positions = self.barrier_engine.get_bound_positions()
    
            if prune_unbound_LEFs:
                LEF_positions = self.extrusion_engine.get_bound_positions()
            else:
                LEF_positions = self.extrusion_engine.get_positions()
                
            self.state_trajectory.append(LEF_states)
    
            self.lef_trajectory.append(LEF_positions)
            self.ctcf_trajectory.append(CTCF_positions)


    def clear_trajectory(self):

        self.state_trajectory = []

        self.lef_trajectory = []
        self.ctcf_trajectory = []
