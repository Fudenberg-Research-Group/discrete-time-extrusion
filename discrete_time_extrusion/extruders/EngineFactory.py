import warnings

from .engines.DiffusionEngines import _diffusion_step_cpu, _diffusion_step_gpu
from .engines.SymmetricEngines import _symmetric_step_cpu, _symmetric_step_gpu
from .engines.AsymmetricEngines import _asymmetric_step_cpu, _asymmetric_step_gpu
	
try:
    import cupy as cp
    
    diffusion_step_cuda = cp.RawKernel(_diffusion_step_gpu(), '_diffusion_step_gpu', options=('--use_fast_math',))
    symmetric_step_cuda = cp.RawKernel(_symmetric_step_gpu(), '_symmetric_step_gpu', options=('--use_fast_math',))
    asymmetric_step_cuda = cp.RawKernel(_asymmetric_step_gpu(), '_asymmetric_step_gpu', options=('--use_fast_math',))
    
    cuda_engines = {'diffusion' : diffusion_step_cuda,
                    'symmetric' : symmetric_step_cuda,
                    'asymmetric' : asymmetric_step_cuda}

except:
    pass
    
try:
    import numba as nb
    use_numba = True

    diffusion_step_numba = nb.njit(fastmath=True)(_diffusion_step_cpu)
    symmetric_step_numba = nb.njit(fastmath=True)(_symmetric_step_cpu)
    asymmetric_step_numba = nb.njit(fastmath=True)(_asymmetric_step_cpu)
    
    numba_engines = {'diffusion' : diffusion_step_numba,
                     'symmetric' : symmetric_step_numba,
                     'asymmetric' : asymmetric_step_numba}

except ImportError:
    use_numba = False

    diffusion_step = _diffusion_step_cpu
    symmetric_step = _symmetric_step_cpu
    asymmetric_step = _asymmetric_step_cpu
    
    python_engines = {'diffusion' : diffusion_step,
                      'symmetric' : symmetric_step,
                      'asymmetric' : asymmetric_step}


def DiffusionEngine(sim, unbound_state_id, threads_per_block=256, **kwargs):

	rngs = sim.xp.random.random((sim.number, 3)).astype(sim.xp.float64)
	args = tuple([unbound_state_id,
				  rngs,
				  sim.number,
				  0, sim.lattice_size,
				  sim.states,
				  sim.occupied,
				  sim.stalled,
				  sim.diffusion_prob,
				  sim.positions])
				  
	if sim.xp.__name__ == 'cupy':
		num_blocks = (sim.number+threads_per_block-1) // threads_per_block
		
		warnings.warn("Running lattice diffusion on the GPU")
		cuda_engines['diffusion']((num_blocks,), (threads_per_block,), args)
		
	elif sim.xp.__name__ == 'numpy':
		if use_numba:
			warnings.warn("Running lattice diffusion on the CPU using Numba")
			numba_engines['diffusion'](*args)
			
		else:
			warnings.warn("Running lattice diffusion on the CPU using pure Python backend")
			python_engines['diffusion'](*args)

	sim.update_occupancies()
			                

def SteppingEngine(sim, mode, active_state_id, threads_per_block=256, **kwargs):
			      
	if mode == "symmetric":
		rngs = sim.xp.random.random((sim.number, 4)).astype(sim.xp.float64)
		args = tuple([active_state_id,
			          rngs,
			          sim.number,
			          0, sim.lattice_size,
			          sim.states,
			          sim.occupied,
			          sim.barrier_engine.stall_left,
			          sim.barrier_engine.stall_right,
			          sim.pause_prob,
			          sim.positions,
			          sim.stalled])

	elif mode == "asymmetric":
		rngs = sim.xp.random.random((sim.number, 2)).astype(sim.xp.float64)
		args = tuple([active_state_id,
			          rngs,
			          sim.number,
			          0, sim.lattice_size,
			          sim.states,
			          sim.occupied,
			          sim.directions,
			          sim.barrier_engine.stall_left,
			          sim.barrier_engine.stall_right,
			          sim.pause_prob,
			          sim.positions,
			          sim.stalled])

	else:
		raise RuntimeError("Unsupported mode '%s'" % mode)
		
	if sim.xp.__name__ == 'cupy':
		num_blocks = (sim.number+threads_per_block-1) // threads_per_block
		
		warnings.warn("Running lattice extrusion on the GPU")
		cuda_engines[mode]((num_blocks,), (threads_per_block,), args)
				
	elif sim.xp.__name__ == 'numpy':
		if use_numba:
			warnings.warn("Running lattice extrusion on the CPU using Numba")
			numba_engines[mode](*args)

		else:
			warnings.warn("Running lattice extrusion on the CPU using pure Python backend")
			python_engines[mode](*args)
			
	sim.update_occupancies()
