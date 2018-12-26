#define LOCAL_DEPTH_2 2
#define LOCAL_DEPTH_4 4
#define LOCAL_DEPTH_8 8
#define LOCAL_DEPTH_16 16
#define LOCAL_DEPTH_32 32
#define LOCAL_HEIGHT 13
#define LOCAL_WIDTH 13

__kernel void fixedpoint2_convolution(__global float *inputs, 	__global float *weights, 	
								__global float *outputs,
								const int K, 				const int C, 		
								const int W, 				const int H,
								const int k_size,  			const int out_w,
								const int out_h, 			const int pad, 	
								const int stride
								)
{
	__local float localImage[15][15];
	__local float localFilter[LOCAL_DEPTH_2][9];
 
	int group_x = get_group_id(0);
	int group_y = get_group_id(1);

	int global_z = get_global_id(2);

	int local_x = get_local_id(0);
	int local_y = get_local_id(1);

	int lxsize = get_local_size(0);
	int lysize = get_local_size(1);

	__private float ans[LOCAL_DEPTH_2] = {0.0}; 

	int c, h, w, d;

	int local_idx = local_y*lxsize + local_x;

	int idx, my_idx;

	for (c=0; c<C; c++) {
		my_idx = c*(H+pad)*(W+pad) + group_y*lysize*(W+pad) + group_x*lxsize;
		for (idx=local_idx; idx<15*15; idx+=13*13) {
			int curr_y = idx/15, curr_x = idx%15;
			localImage[curr_y][curr_x] = inputs[my_idx + curr_y*(W+pad) + curr_x];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		for (idx=local_idx; idx<LOCAL_DEPTH_2*9; idx+=13*13) {
			int chunk_num = idx/9, pos = idx%9;
			localFilter[chunk_num][pos] = weights[(global_z*LOCAL_DEPTH_2 + chunk_num)*C*9 + c*9 + pos];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		for (d=0; d<LOCAL_DEPTH_2; d++) {
			for (h=0; h<3; h++) {
				for (w=0; w<3; w++) {
					ans[d] += (localFilter[d][k_size*h+w] * localImage[local_y+h][local_x+w]);
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


__kernel void fixedpoint4_convolution(__global float *inputs, 	__global float *weights, 	
								__global float *outputs,
								const int K, 				const int C, 		
								const int W, 				const int H,
								const int k_size,  			const int out_w,
								const int out_h, 			const int pad, 	
								const int stride
								)
{
	__local float localImage[15][15];
	__local float localFilter[LOCAL_DEPTH_4][9];
 
	int group_x = get_group_id(0);
	int group_y = get_group_id(1);

	int global_z = get_global_id(2);

	int local_x = get_local_id(0);
	int local_y = get_local_id(1);

	int lxsize = get_local_size(0);
	int lysize = get_local_size(1);

	__private float ans[LOCAL_DEPTH_4] = {0.0}; 

	int c, h, w, d;

	int local_idx = local_y*lxsize + local_x;

	int idx, my_idx;

	for (c=0; c<C; c++) {
		my_idx = c*(H+pad)*(W+pad) + group_y*lysize*(W+pad) + group_x*lxsize;
		for (idx=local_idx; idx<15*15; idx+=13*13) {
			int curr_y = idx/15, curr_x = idx%15;
			localImage[curr_y][curr_x] = inputs[my_idx + curr_y*(W+pad) + curr_x];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		for (idx=local_idx; idx<LOCAL_DEPTH_4*9; idx+=13*13) {
			int chunk_num = idx/9, pos = idx%9;
			localFilter[chunk_num][pos] = weights[(global_z*LOCAL_DEPTH_4 + chunk_num)*C*9 + c*9 + pos];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		for (d=0; d<LOCAL_DEPTH_4; d++) {
			for (h=0; h<3; h++) {
				for (w=0; w<3; w++) {
					ans[d] += (localFilter[d][k_size*h+w] * localImage[local_y+h][local_x+w]);
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

__kernel void fixedpoint8_convolution(__global float *inputs, 	__global float *weights, 	
								__global float *outputs,
								const int K, 				const int C, 		
								const int W, 				const int H,
								const int k_size,  			const int out_w,
								const int out_h, 			const int pad, 	
								const int stride
								)
{
	__local float localImage[15][15];
	__local float localFilter[LOCAL_DEPTH_8][9];
 
	int group_x = get_group_id(0);
	int group_y = get_group_id(1);

	int global_z = get_global_id(2);

	int local_x = get_local_id(0);
	int local_y = get_local_id(1);

	int lxsize = get_local_size(0);
	int lysize = get_local_size(1);

	__private float ans[LOCAL_DEPTH_8] = {0.0}; 

	int c, h, w, d;

	int local_idx = local_y*lxsize + local_x;

	int idx, my_idx;

	for (c=0; c<C; c++) {
		my_idx = c*(H+pad)*(W+pad) + group_y*lysize*(W+pad) + group_x*lxsize;
		for (idx=local_idx; idx<15*15; idx+=13*13) {
			int curr_y = idx/15, curr_x = idx%15;
			localImage[curr_y][curr_x] = inputs[my_idx + curr_y*(W+pad) + curr_x];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		for (idx=local_idx; idx<LOCAL_DEPTH_8*9; idx+=13*13) {
			int chunk_num = idx/9, pos = idx%9;
			localFilter[chunk_num][pos] = weights[(global_z*LOCAL_DEPTH_8 + chunk_num)*C*9 + c*9 + pos];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		for (d=0; d<LOCAL_DEPTH_8; d++) {
			for (h=0; h<3; h++) {
				for (w=0; w<3; w++) {
					ans[d] += (localFilter[d][k_size*h+w] * localImage[local_y+h][local_x+w]);
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

__kernel void fixedpoint16_convolution(__global float *inputs, 	__global float *weights, 	
								__global float *outputs,
								const int K, 				const int C, 		
								const int W, 				const int H,
								const int k_size,  			const int out_w,
								const int out_h, 			const int pad, 	
								const int stride
								)
{
	__local float localImage[15][15];
	__local float localFilter[LOCAL_DEPTH_16][9];
 
	int group_x = get_group_id(0);
	int group_y = get_group_id(1);

	int global_z = get_global_id(2);

	int local_x = get_local_id(0);
	int local_y = get_local_id(1);

	int lxsize = get_local_size(0);
	int lysize = get_local_size(1);

	__private float ans[LOCAL_DEPTH_16] = {0.0}; 

	int c, h, w, d;

	int local_idx = local_y*lxsize + local_x;

	int idx, my_idx;

	for (c=0; c<C; c++) {
		my_idx = c*(H+pad)*(W+pad) + group_y*lysize*(W+pad) + group_x*lxsize;
		for (idx=local_idx; idx<15*15; idx+=13*13) {
			int curr_y = idx/15, curr_x = idx%15;
			localImage[curr_y][curr_x] = inputs[my_idx + curr_y*(W+pad) + curr_x];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		for (idx=local_idx; idx<LOCAL_DEPTH_16*9; idx+=13*13) {
			int chunk_num = idx/9, pos = idx%9;
			localFilter[chunk_num][pos] = weights[(global_z*LOCAL_DEPTH_16 + chunk_num)*C*9 + c*9 + pos];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		for (d=0; d<LOCAL_DEPTH_16; d++) {
			for (h=0; h<3; h++) {
				for (w=0; w<3; w++) {
					ans[d] += (localFilter[d][k_size*h+w] * localImage[local_y+h][local_x+w]);
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

__kernel void fixedpoint32_convolution(__global float *inputs, 	__global float *weights, 	
								__global float *outputs,
								const int K, 				const int C, 		
								const int W, 				const int H,
								const int k_size,  			const int out_w,
								const int out_h, 			const int pad, 	
								const int stride
								)
{
	__local float localImage[15][15];
	__local float localFilter[LOCAL_DEPTH_32][9];
 
	int group_x = get_group_id(0);
	int group_y = get_group_id(1);

	int global_z = get_global_id(2);

	int local_x = get_local_id(0);
	int local_y = get_local_id(1);

	int lxsize = get_local_size(0);
	int lysize = get_local_size(1);

	__private float ans[LOCAL_DEPTH_32] = {0.0}; 

	int c, h, w, d;

	int local_idx = local_y*lxsize + local_x;

	int idx, my_idx;

	for (c=0; c<C; c++) {
		my_idx = c*(H+pad)*(W+pad) + group_y*lysize*(W+pad) + group_x*lxsize;
		for (idx=local_idx; idx<15*15; idx+=13*13) {
			int curr_y = idx/15, curr_x = idx%15;
			localImage[curr_y][curr_x] = inputs[my_idx + curr_y*(W+pad) + curr_x];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		for (idx=local_idx; idx<LOCAL_DEPTH_32*9; idx+=13*13) {
			int chunk_num = idx/9, pos = idx%9;
			localFilter[chunk_num][pos] = weights[(global_z*LOCAL_DEPTH_32 + chunk_num)*C*9 + c*9 + pos];
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		for (d=0; d<LOCAL_DEPTH_32; d++) {
			for (h=0; h<3; h++) {
				for (w=0; w<3; w++) {
					ans[d] += (localFilter[d][k_size*h+w] * localImage[local_y+h][local_x+w]);
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
