def _symmetric_step_cpu(active_state_id,
                        rngs,
                        N,
                        N_min,
                        N_max,
                        states,
                        occupied,
                        stall_left,
                        stall_right,
                        pause_prob,
                        positions,
                        stalled):
					
	for i in range(N):
		if states[i] == active_state_id:
			cur1 = positions[i, 0]
			cur2 = positions[i, 1]
			
			stall1 = stall_left[cur1]
			stall2 = stall_right[cur2]
									
			if rngs[i, 0] < stall1:
				stalled[i, 0] = True
				
			if rngs[i, 1] < stall2:
				stalled[i, 1] = True

			if not stalled[i, 0]:
				if not occupied[cur1-1]:
					pause1 = pause_prob[cur1]
					
					if rngs[i, 2] > pause1:
						positions[i, 0] = cur1 - 1
						
			if not stalled[i, 1]:
				if not occupied[cur2+1]:
					pause2 = pause_prob[cur2]
					
					if rngs[i, 3] > pause2:
						positions[i, 1] = cur2 + 1


def _symmetric_step_gpu():

	return r'''
	
	extern "C"
	__global__ void _symmetric_step_gpu(
			const int active_state_id,
			const double* rngs,
			const unsigned int N,
			const unsigned int N_min,
			const unsigned int N_max,
			const int* states,
			const bool* occupied,
			const double* stall_left,
			const double* stall_right,
			const double* pause_prob,
			int* positions,
			unsigned int* stalled) {

		unsigned int i = (unsigned int) (blockDim.x * blockIdx.x + threadIdx.x);

		if (i >= N)
			return;
			
		if (states[i] != active_state_id)
			return;

		uint2* stall = (uint2*) stalled;
		int2* position = (int2*) positions;
		
		double4* rng = (double4*) rngs;

		unsigned int cur1 = (unsigned int) position[i].x;
		unsigned int cur2 = (unsigned int) position[i].y;

		double stall1 = stall_left[cur1];
		double stall2 = stall_right[cur2];
								
		if (rng[i].w < stall1)
			stall[i].x = 1;
			
		if (rng[i].x < stall2)
			stall[i].y = 1;
						
		if (stall[i].x == 0) {
			if (!occupied[cur1-1]) {
				double pause1 = pause_prob[cur1];
				
				if (rng[i].y > pause1)
					position[i].x = (int) cur1-1;
			}
		}
					
		if (stall[i].y == 0) {
			if (!occupied[cur2+1]) {
				double pause2 = pause_prob[cur2];
				
				if (rng[i].z > pause2)
					position[i].y = (int) cur2+1;
			}
		}
	}
	'''
