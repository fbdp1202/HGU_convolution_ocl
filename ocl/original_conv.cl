__kernel void image_convolution(__global float *inputs, 	__global float *weights, 	
								__global float *outputs,	
								const int K, 				const int C, 		
								const int W, 				const int H,
								const int k_size,  			const int out_w,
								const int out_h, 			const int pad, 	
								const int stride
								)
{
	int k, c, w, h, mh, mw;
	int outpus_size = out_w * out_h;
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
							int weightIdx = ((k*C+c)*k_size+mh)*k_size+mw;
							ret += inputs[inputIdx] * weights[weightIdx];
						}
					}
				}
				outputs[k*outpus_size + h*out_w + w] = ret;
			}
		}
	}
}