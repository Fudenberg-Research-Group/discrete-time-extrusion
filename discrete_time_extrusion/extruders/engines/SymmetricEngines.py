def _symmetric_step_cpu(active_state_id,
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
                        occupied):
					
    for i in range(N):
        if states[i] == active_state_id:
            for leg_id in range(2):
                head = 2*(i+leg_id)*(step_size+2)

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
                                    positions[i, 0] = new_pos - 1
        					    	
                                    occupied[new_pos] = 0 if positions[i, 1] != new_pos else 1
                                    occupied[new_pos-1] = 1
                                    
                                else:
                                    break
        						    
                            elif leg_id == 1:
                                if not occupied[new_pos+1]:
                                    positions[i, 1] = new_pos + 1
        					    	
                                    occupied[new_pos] = 0 if positions[i, 0] != new_pos else 1
                                    occupied[new_pos+1] = 1

                                else:
                                    break


def _symmetric_step_gpu():

	return r'''
	
	extern "C"
	__global__ void _symmetric_step_gpu(
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
			unsigned int* occupied) {

		unsigned int i = (unsigned int) (blockDim.x * blockIdx.x + threadIdx.x);

		if (i >= N)
			return;
			
		if (states[i] != active_state_id)
			return;

		uint2* stall = (uint2*) stalled;
		int2* position = (int2*) positions;
		
		double2* forward_rate = (double2*) forward_rates;

		unsigned int head = 2*i*(step_size+2);
		
		unsigned int old_pos1 = (unsigned int) position[i].x;
		unsigned int old_pos2 = (unsigned int) position[i].y;

		double stall1 = stall_left[old_pos1];
		double stall2 = stall_right[old_pos2];

		double step_rate1 = step_prob[old_pos1] * forward_rate[i].x;
		double step_rate2 = step_prob[old_pos2] * forward_rate[i].y;
		
		if (rngs[head++] < stall1) 
			stall[i].x = 1;
		if (rngs[head++] < stall2) 
			stall[i].y = 1;
				
		if ( (rngs[head++] < step_rate1) && (stall[i].x == 0) ) {
			for (unsigned int j = 0; j < step_size; ++j) {
				double new_pos = (unsigned int) position[i].x;
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
			
		if ( (rngs[head++] < step_rate2) && (stall[i].y == 0) ) {
			for (unsigned int j = 0; j < step_size; ++j) {
				double new_pos = (unsigned int) position[i].y;
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
	'''
