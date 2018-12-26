__kernel void local_convolution(__global float *inputs, 	__global float *weights, 	
								__global float *outputs,
								const int K, 				const int C, 		
								const int W, 				const int H,
								const int k_size,  			const int out_w,
								const int out_h, 			const int pad, 	
								const int stride,
								__local float *localFilter
								)
{
	int k, c, w, h, mh, mw;
	int outpus_size = out_w * out_h;

	for (k=0; k<K*C*k_size*k_size; k++){
		localFilter[k] = weights[k];
	}

	for (k=0; k<K; k++) {
		for (h=0; h<out_h; h++) {
			for (w=0; w<out_w; w++) {
				float ret = 0;
				for (c=0; c<C; c++) {
					for (mh=0; mh<k_size; mh++) {
						int curr_h = h * stride + mh - pad;
						if (curr_h < 0 || curr_h >= H) continue;
						for (mw=0; mw<k_size; mw++) {
							int curr_w = w * stride + mw - pad;
							if (curr_w < 0 || curr_w >= W) continue;
							int inputIdx = (c*H+curr_h)*W+curr_w;
							ret += inputs[inputIdx] * localFilter[(k*C+c)*k_size*k_size+mh*k_size+mw];
						}
					}
				}
				outputs[k*outpus_size + h*out_w + w] = ret;
			}
		}
	}
}