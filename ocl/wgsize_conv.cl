__kernel void wgsize_convolution(__global float *inputs, 	__global float *weights, 	
								__global float *outputs,
								const int K, 				const int C, 		
								const int W, 				const int H,
								const int k_size,  			const int out_w,
								const int out_h, 			const int pad, 	
								const int stride,
								__local float *localFilter
								)
{
	int global_x = get_global_id(0);
	int global_y = get_global_id(1);

	int gxsize = get_global_size(0);
	int gysize = get_global_size(1);

	int w_start = global_x * gxsize;
	int w_end = w_start + gxsize;
	int h_start = global_y * gysize;
	int h_end = h_start + gysize;

	int k, c, w, h, mh, mw, m;
	int outpus_size = out_w * out_h;
	int Kfilter_bias = 0, Cfilter_bias = 0;

	for (k=0; k<K*C*k_size*k_size; k++){
		localFilter[k] = weights[k];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for (k=0; k<K; k++) {
		for (h=h_start; h<h_end; h++) {
			int h_bias = h*stride;
			for (w=w_start; w<w_end; w++) {
				float ret = 0;
				int w_bias = w*stride;
				Cfilter_bias = Kfilter_bias;
				for (c=0; c<C; c++) {
					for (m=0, mh=0; mh<k_size; mh++) {
						int curr_h = h_bias + mh - pad;
						int curr_h_bias = (c*H+curr_h)*W;
						if (curr_h < 0 || curr_h >= H) continue;
						for (mw=0; mw<k_size; mw++, m++) {
							int curr_w = w_bias + mw - pad;
							if (curr_w < 0 || curr_w >= W) continue;
							int inputIdx = curr_h_bias + curr_w;
							ret += inputs[inputIdx] * localFilter[Cfilter_bias+m];
						}
					}
					Cfilter_bias += k_size*k_size;
				}
				outputs[k*outpus_size + h*out_w + w] = ret;
			}
		}
		Kfilter_bias = Cfilter_bias;
	}
}