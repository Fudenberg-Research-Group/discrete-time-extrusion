def _asymmetric_step_cpu(active_state_id,
                         rngs,
                         N,
                         step_size,
                         states,
                         stall_left,
                         stall_right,
                         step_prob,
                         pause_prob,
                         forward_rates,
                         positions,
                         stalled,
                         directions,
                         occupied):
					
	for i in range(N):
		if states[i] == active_state_id:
			head = i*(step_size+2)
		
			leg_id = directions[i]
			old_pos = positions[i, leg_id]
			
			step_rate = step_prob[old_pos] * forward_rates[i, leg_id]
			stall_rate = stall_left[old_pos] if leg_id == 0 else stall_right[old_pos]

			if rngs[head] < stall_rate:
				stalled[i, leg_id] = True
						
			if (not stalled[i, leg_id]) & (rngs[head+1] < step_rate):
				for j in range(step_size):
					new_pos = positions[i, leg_id]
					pause = pause_prob[new_pos]

					if rngs[head+2+j] > pause:
						if leg_id == 0:
							if not occupied[new_pos-1]:
								positions[i, leg_id] = new_pos - 1
								
								occupied[new_pos] = 0 if positions[i, 1] != new_pos else 1
								occupied[new_pos-1] = 1
								
							else:
								break

						elif leg_id == 1:
							if not occupied[new_pos+1]:
								positions[i, leg_id] = new_pos + 1
								
								occupied[new_pos] = 0 if positions[i, 0] != new_pos else 1
								occupied[new_pos+1] = 1
								
							else:
								break


def _asymmetric_step_gpu():

	return r'''
	
	extern "C"
	__global__ void _asymmetric_step_gpu(
			const int active_state_id,
			const double* rngs,
			const unsigned int N,
			const unsigned int step_size,
			const int* states,
			const double* stall_left,
			const double* stall_right,
			const double* step_prob,
			const double* pause_prob,
			const double* forward_rates,
			int* positions,
			unsigned int* stalled,
			unsigned int* directions,
			unsigned int* occupied) {

		unsigned int i = (unsigned int) (blockDim.x * blockIdx.x + threadIdx.x);

		if (i >= N)
			return;
			
		if (states[i] != active_state_id)
			return;

		uint2* stall = (uint2*) stalled;
		int2* position = (int2*) positions;
		
		double2* forward_rate = (double2*) forward_rates;

		unsigned int leg_id = directions[i];
		unsigned int head = i*(step_size+2);
		
		unsigned int old_pos = (unsigned int) ((leg_id == 0) ? position[i].x : position[i].y);
		
		double step_rate = step_prob[old_pos] * ((leg_id == 0) ? forward_rate[i].x : forward_rate[i].y);
		
		if (rngs[head++] < ((leg_id == 0) ? stall_left[old_pos] : stall_right[old_pos])) {
			if (leg_id == 0)
				stall[i].x = 1;
			else if (leg_id == 1)
				stall[i].y = 1;
		}
				
		if (leg_id == 0) {
			if ( (rngs[head++] < step_rate) && (stall[i].x == 0) ) {
				for (unsigned int j = 0; j < step_size; ++j) {
					unsigned int new_pos = (unsigned int) position[i].x;
					double pause = pause_prob[new_pos];

					if (rngs[head++] > pause) {
						if (atomicCAS(&occupied[new_pos-1], 0, 1) == 0) {
							if (position[i].y != new_pos)
								atomicExch(&occupied[new_pos], 0);
								
							position[i].x = (int) new_pos-1;
						}
						
						else 
						    break;
					}
				}
			}
		}
							
		else if (leg_id == 1) {
			if ( (rngs[head++] < step_rate) && (stall[i].y == 0) ) {
				for (unsigned int j = 0; j < step_size; ++j) {
					unsigned int new_pos = (unsigned int) position[i].y;
					double pause = pause_prob[new_pos];
					
					if (rngs[head++] > pause) {
						if (atomicCAS(&occupied[new_pos+1], 0, 1) == 0) {
							if (position[i].x != new_pos)
								atomicExch(&occupied[new_pos], 0);
								
							position[i].y = (int) new_pos+1;
						}
						
						else 
						    break;
					}
				}
			}
		}
	}
	'''
