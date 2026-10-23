def _diffusion_step_cpu(unbound_state_id,
                        rngs,
                        N,
                        states,
                        diffuse_prob,
                        forward_rates,
                        backward_rates,
                        positions,
                        stalled,
                        occupied):
					
	for i in range(N):
		if (states[i] != unbound_state_id):
			leg_id = 0 if rngs[i, 0] < 0.5 else 1
			old_pos = positions[i, leg_id]
			
			if (leg_id == 0):
			    if (rngs[i, 1] < 0.5):
			        rate = forward_rates[i, 0]
			    else:
			        rate = backward_rates[i, 0]

			else:
			    if (rngs[i, 1] < 0.5):
			        rate = backward_rates[i, 1]
			    else:
			        rate = forward_rates[i, 1]
			        
			if rngs[i, 2] < diffuse_prob[old_pos] * rate:
				stalled[i, leg_id] = False
				
				if rngs[i, 1] < 0.5:
					if (not occupied[old_pos-1]) and ((leg_id == 0) or (old_pos > positions[i, 0])):
						positions[i, leg_id] = old_pos - 1
							
						occupied[old_pos-1] = 1
						occupied[old_pos] = 0 if positions[i, 1-leg_id] != old_pos else 1

				else:
					if (not occupied[old_pos+1]) and ((leg_id == 1) or (old_pos < positions[i, 1])):
						positions[i, leg_id] = old_pos + 1
							
						occupied[old_pos+1] = 1
						occupied[old_pos] = 0 if positions[i, 1-leg_id] != old_pos else 1


def _diffusion_step_gpu():

	return r'''
	
	extern "C"
	__global__ void _diffusion_step_gpu(
			const int unbound_state_id,
			const double* rngs,
			const unsigned int N,
			const int* states,
			const double* diffuse_prob,
			const double* forward_rates,
			const double* backward_rates,
			int* positions,
			unsigned int* stalled,
			unsigned int* occupied) {

		unsigned int i = (unsigned int) (blockDim.x * blockIdx.x + threadIdx.x);

		if (i >= N)
			return;
			
		if (states[i] == unbound_state_id)
			return;

		uint2* stall = (uint2*) stalled;
		int2* position = (int2*) positions;
		
		double3* rng = (double3*) rngs;
		
		double2* forward_rate = (double2*) forward_rates;
		double2* backward_rate = (double2*) backward_rates;
		
		unsigned int old_pos1 = (unsigned int) position[i].x;
		unsigned int old_pos2 = (unsigned int) position[i].y;
		
		if (rng[i].x < 0.5) {
			double rate = (rng[i].y < 0.5) ? forward_rate[i].x : backward_rate[i].x;

			if (rng[i].z < diffuse_prob[old_pos1] * rate) {
				stall[i].x = 0;
			
				if (rng[i].y < 0.5) {
					if (atomicCAS(&occupied[old_pos1-1], 0, 1) == 0) {
						if (position[i].y != old_pos1)
							atomicExch(&occupied[old_pos1], 0);
							
						position[i].x = (int) old_pos1-1;
					}
				}
				
				else {
					if (old_pos1 < position[i].y) {
						if (atomicCAS(&occupied[old_pos1+1], 0, 1) == 0) {
							atomicExch(&occupied[old_pos1], 0);
							
							position[i].x = (int) old_pos1+1;
						}
					}
				}
			}
		}
		
		else {
			double rate = (rng[i].y < 0.5) ? backward_rate[i].y : forward_rate[i].y;

			if (rng[i].z < diffuse_prob[old_pos2] * rate) {
				stall[i].y = 0;

				if (rng[i].y < 0.5) {
					if (old_pos2 > position[i].x) {
						if (atomicCAS(&occupied[old_pos2-1], 0, 1) == 0) {
							atomicExch(&occupied[old_pos2], 0);
							
							position[i].y = (int) old_pos2-1;
						}
					}
				}
				
				else {
					if (atomicCAS(&occupied[old_pos2+1], 0, 1) == 0) {
						if (position[i].x != old_pos2)
							atomicExch(&occupied[old_pos2], 0);
							
						position[i].y = (int) old_pos2+1;
					}
				}
			}
		}
	}
	'''
