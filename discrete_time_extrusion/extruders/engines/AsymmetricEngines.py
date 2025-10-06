def _asymmetric_step_cpu(active_state_id,
                         rngs,
                         N,
                         N_min,
                         N_max,
                         states,
                         occupied,
                         directions,
                         stall_left,
                         stall_right,
                         pause_prob,
                         positions,
                         stalled):
					
	for i in range(N):
		if states[i] == active_state_id:
			leg_id = directions[i]
			
			cur = positions[i, leg_id]
			stall = stall_left[cur] if leg_id == 0 else stall_right[cur]
									
			if rngs[i, 0] < stall:
				stalled[i, leg_id] = True

			if not stalled[i, leg_id]:
				pause = pause_prob[cur]

				if leg_id == 0:
					if not occupied[cur-1]:
						if rngs[i, 1] > pause:
							positions[i, leg_id] = cur - 1

				elif leg_id == 1:
					if not occupied[cur+1]:
						if rngs[i, 1] > pause:
							positions[i, leg_id] = cur + 1


def _asymmetric_step_gpu():

	return r'''
	
	extern "C"
	__global__ void _asymmetric_step_gpu(
			const int active_state_id,
			const double* rngs,
			const unsigned int N,
			const unsigned int N_min,
			const unsigned int N_max,
			const int* states,
			const bool* occupied,
			const unsigned int* directions,
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
		
		double2* rng = (double2*) rngs;

		unsigned int leg_id = directions[i];
		unsigned int cur = (unsigned int) (leg_id==0 ? position[i].x : position[i].y);
			
		if (rng[i].x < (leg_id==0 ? stall_left[cur] : stall_right[cur])) {
			if (leg_id == 0)
				stall[i].x = 1;
			else if (leg_id == 1)
				stall[i].y = 1;
		}
		
		if (leg_id == 0) {
			if (stall[i].x == 0) {
				if (!occupied[cur-1]) {
					if (rng[i].y > pause_prob[cur])
						position[i].x = (int) cur-1;
				}
			}
		}
					
		else if (leg_id == 1) {
			if (stall[i].y == 0) {
				if (!occupied[cur+1]) {
					if (rng[i].y > pause_prob[cur])
						position[i].y = (int) cur+1;
				}
			}
		}
	}
	'''
