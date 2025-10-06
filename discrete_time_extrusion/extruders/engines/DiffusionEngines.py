def _diffusion_step_cpu(unbound_state_id,
                        rngs,
                        N,
                        N_min,
                        N_max,
                        states,
                        occupied,
                        stalled,
                        diffuse_prob,
                        positions):
					
	for i in range(N):
		if (states[i] != unbound_state_id):
			for j in range(2):
				cur = positions[i, j]

				if rngs[i, 2*j] < diffuse_prob[cur]:
					stalled[i, j] = False
				
					if rngs[i, 2*j+1] < 0.5:
						if (not occupied[cur-1]) and ((j == 0) or (cur > positions[i, 0])):
							positions[i, j] = cur - 1
							
					else:
						if (not occupied[cur+1]) and ((j == 1) or (cur < positions[i, 1])):
							positions[i, j] = cur + 1


def _diffusion_step_gpu():

	return r'''
	
	extern "C"
	__global__ void _diffusion_step_gpu(
			const int unbound_state_id,
			const double* rngs,
			const unsigned int N,
			const unsigned int N_min,
			const unsigned int N_max,
			const int* states,
			const bool* occupied,
			const unsigned int* stalled,
			const double* diffuse_prob,
			int* positions) {

		unsigned int i = (unsigned int) (blockDim.x * blockIdx.x + threadIdx.x);

		if (i >= N)
			return;
			
		if (states[i] == unbound_state_id)
			return;

		uint2* stall = (uint2*) stalled;
		int2* position = (int2*) positions;
		
		double3* rng = (double3*) rngs;

		unsigned int cur1 = (unsigned int) position[i].x;
		unsigned int cur2 = (unsigned int) position[i].y;

		if (rng[i].x < 0.5) {
			if (rng[i].y < diffuse_prob[cur1]) {
				stall[i].x = 0;
			
				if (rng[i].z < 0.5) {
					if (!occupied[cur1-1])
						position[i].x = (int) cur1-1;
				}
				else {
					if (!occupied[cur1+1] && (cur1 < position[i].y))
						position[i].x = (int) cur1+1;
				}
			}
		}
		
		else {
			if (rng[i].y < diffuse_prob[cur2]) {
				stall[i].y = 0;

				if (rng[i].z < 0.5) {
					if (!occupied[cur2-1] && (cur2 > position[i].x))
						position[i].y = (int) cur2-1;
				}
				else {
					if (!occupied[cur2+1])
						position[i].y = (int) cur2+1;
				}
			}
		}
	}
	'''
