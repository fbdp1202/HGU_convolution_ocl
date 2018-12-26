#ifdef OPENCL
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "layer.h"
#include "utils.h"
#include "convolution.h"
#include "define_cl.h"

void init_ocl_convolution(layer *pl, float *inputs) {
	int input_size = pl->c * pl->w * pl->h;
	int output_size = pl->k * pl->out_w * pl->out_h;
	int weight_size = pl->k * pl->c * pl->size * pl->size;

	pl->mo_inputs = clCreateMemobj(CL_MEM_READ_ONLY, sizeof(float) * input_size, NULL);
	pl->mo_weights = clCreateMemobj(CL_MEM_READ_ONLY, sizeof(float) * weight_size, NULL);
	pl->mo_outputs = clCreateMemobj(CL_MEM_READ_WRITE, sizeof(float) * output_size, NULL);

	if (pl->pad) {
		int pad_input_size = pl->c * (pl->w + pl->pad) * (pl->h + pl->pad);
		float *pad_inputs = calloc(pad_input_size, sizeof(float));
		int w, h, c;
		for (c = 0; c < pl->c; c++) {
			for (h = 0; h < pl->h; h++) {
				for (w = 0; w < pl->w; w++) {
					int input_idx = (c*pl->h+h)*pl->w+w;
					int pad_idx = (c*pl->h+h+pl->pad)*pl->w+w+pl->pad;
					pad_inputs[pad_idx] = inputs[input_idx];
				}
			}
		}
		pl->mo_pad_inputs = clCreateMemobj(CL_MEM_READ_ONLY, sizeof(float) * pad_input_size, NULL);
		cl_memcpy_to_device(pl->mo_pad_inputs, pad_inputs, sizeof(float) * pad_input_size);
	}

	cl_memcpy_to_device(pl->mo_inputs, inputs, sizeof(float) * input_size);
	cl_memcpy_to_device(pl->mo_weights, pl->weights, sizeof(float) * weight_size);
}

void free_ocl(layer *pl) {
	if (pl->mo_pad_inputs) 	clFreeMemobj(pl->mo_pad_inputs);
	if (pl->mo_inputs) 		clFreeMemobj(pl->mo_inputs);
	if (pl->mo_weights) 	clFreeMemobj(pl->mo_weights);
	if (pl->mo_outputs) 	clFreeMemobj(pl->mo_outputs);
}

// -----------------------------------------------------------------------------------------------------------------------//

void original_image_convolution(layer *pl, float *inputs) {
	int output_size = pl->k * pl->out_w * pl->out_h;

	size_t global[3] = { 1, 1, 1 };
	size_t local[3] ={ 1, 1, 1 };
	cl_kernel krnl_to_execute = clGetkrnl_orig_conv();

    clSetKernelArg(krnl_to_execute, 0, sizeof(cl_mem), &pl->mo_inputs);
    clSetKernelArg(krnl_to_execute, 1, sizeof(cl_mem), &pl->mo_weights);
    clSetKernelArg(krnl_to_execute, 2, sizeof(cl_mem), &pl->mo_outputs);
    clSetKernelArg(krnl_to_execute, 3, sizeof(int), &pl->k);
    clSetKernelArg(krnl_to_execute, 4, sizeof(int), &pl->c);
    clSetKernelArg(krnl_to_execute, 5, sizeof(int), &pl->w);
    clSetKernelArg(krnl_to_execute, 6, sizeof(int), &pl->h);
    clSetKernelArg(krnl_to_execute, 7, sizeof(int), &pl->size);
    clSetKernelArg(krnl_to_execute, 8, sizeof(int), &pl->out_w);
    clSetKernelArg(krnl_to_execute, 9, sizeof(int), &pl->out_h);
    clSetKernelArg(krnl_to_execute, 10, sizeof(int), &pl->pad);
    clSetKernelArg(krnl_to_execute, 11, sizeof(int), &pl->stride);

    cl_run_kernel3d(krnl_to_execute, global, local, 3);

	cl_memcpy_from_device(pl->outputs, pl->mo_outputs, sizeof(float) * output_size);
}

// -----------------------------------------------------------------------------------------------------------------------//

void local_image_convolution(layer *pl, float *inputs) {
	int output_size = pl->k * pl->out_w * pl->out_h;
	int weight_size = pl->k*pl->c*pl->size*pl->size;

	size_t global[3] = { 1, 1, 1 };
	size_t local[3] ={ 1, 1, 1 };

	cl_kernel krnl_to_execute = clGetkrnl_local_conv();

    clSetKernelArg(krnl_to_execute, 0, sizeof(cl_mem), &pl->mo_inputs);
    clSetKernelArg(krnl_to_execute, 1, sizeof(cl_mem), &pl->mo_weights);
    clSetKernelArg(krnl_to_execute, 2, sizeof(cl_mem), &pl->mo_outputs);
    clSetKernelArg(krnl_to_execute, 3, sizeof(int), &pl->k);
    clSetKernelArg(krnl_to_execute, 4, sizeof(int), &pl->c);
    clSetKernelArg(krnl_to_execute, 5, sizeof(int), &pl->w);
    clSetKernelArg(krnl_to_execute, 6, sizeof(int), &pl->h);
    clSetKernelArg(krnl_to_execute, 7, sizeof(int), &pl->size);
    clSetKernelArg(krnl_to_execute, 8, sizeof(int), &pl->out_w);
    clSetKernelArg(krnl_to_execute, 9, sizeof(int), &pl->out_h);
    clSetKernelArg(krnl_to_execute, 10, sizeof(int), &pl->pad);
    clSetKernelArg(krnl_to_execute, 11, sizeof(int), &pl->stride);
    clSetKernelArg(krnl_to_execute, 12, weight_size*sizeof(float), NULL);

    cl_run_kernel3d(krnl_to_execute, global, local, 3);

	cl_memcpy_from_device(pl->outputs, pl->mo_outputs, sizeof(float) * output_size);
}

// -----------------------------------------------------------------------------------------------------------------------//

void memorize_image_convolution(layer *pl, float *inputs) {
	int output_size = pl->k * pl->out_w * pl->out_h;
	int weight_size = pl->k*pl->c*pl->size*pl->size;
	size_t global[3] = { 1, 1, 1 };
	size_t local[3] ={ 1, 1, 1 };

	cl_kernel krnl_to_execute = clGetkrnl_memorize_conv();

    clSetKernelArg(krnl_to_execute, 0, sizeof(cl_mem), &pl->mo_inputs);
    clSetKernelArg(krnl_to_execute, 1, sizeof(cl_mem), &pl->mo_weights);
    clSetKernelArg(krnl_to_execute, 2, sizeof(cl_mem), &pl->mo_outputs);
    clSetKernelArg(krnl_to_execute, 3, sizeof(int), &pl->k);
    clSetKernelArg(krnl_to_execute, 4, sizeof(int), &pl->c);
    clSetKernelArg(krnl_to_execute, 5, sizeof(int), &pl->w);
    clSetKernelArg(krnl_to_execute, 6, sizeof(int), &pl->h);
    clSetKernelArg(krnl_to_execute, 7, sizeof(int), &pl->size);
    clSetKernelArg(krnl_to_execute, 8, sizeof(int), &pl->out_w);
    clSetKernelArg(krnl_to_execute, 9, sizeof(int), &pl->out_h);
    clSetKernelArg(krnl_to_execute, 10, sizeof(int), &pl->pad);
    clSetKernelArg(krnl_to_execute, 11, sizeof(int), &pl->stride);
    clSetKernelArg(krnl_to_execute, 12, weight_size*sizeof(float), NULL);

    cl_run_kernel3d(krnl_to_execute, global, local, 3);

	cl_memcpy_from_device(pl->outputs, pl->mo_outputs, sizeof(float) * output_size);
}

// -----------------------------------------------------------------------------------------------------------------------//

void wgsize_image_convolution(layer *pl, float *inputs) {
	int output_size = pl->k * pl->out_w * pl->out_h;
	int weight_size = pl->k*pl->c*pl->size*pl->size;
	size_t global[3] = { 13, 13, 1 };
	size_t local[3] ={ 13, 13, 1 };

	cl_kernel krnl_to_execute = clGetkrnl_wgsize_conv();

    clSetKernelArg(krnl_to_execute, 0, sizeof(cl_mem), &pl->mo_inputs);
    clSetKernelArg(krnl_to_execute, 1, sizeof(cl_mem), &pl->mo_weights);
    clSetKernelArg(krnl_to_execute, 2, sizeof(cl_mem), &pl->mo_outputs);
    clSetKernelArg(krnl_to_execute, 3, sizeof(int), &pl->k);
    clSetKernelArg(krnl_to_execute, 4, sizeof(int), &pl->c);
    clSetKernelArg(krnl_to_execute, 5, sizeof(int), &pl->w);
    clSetKernelArg(krnl_to_execute, 6, sizeof(int), &pl->h);
    clSetKernelArg(krnl_to_execute, 7, sizeof(int), &pl->size);
    clSetKernelArg(krnl_to_execute, 8, sizeof(int), &pl->out_w);
    clSetKernelArg(krnl_to_execute, 9, sizeof(int), &pl->out_h);
    clSetKernelArg(krnl_to_execute, 10, sizeof(int), &pl->pad);
    clSetKernelArg(krnl_to_execute, 11, sizeof(int), &pl->stride);
    clSetKernelArg(krnl_to_execute, 12, weight_size*sizeof(float), NULL);

    cl_run_kernel3d(krnl_to_execute, global, local, 3);

	cl_memcpy_from_device(pl->outputs, pl->mo_outputs, sizeof(float) * output_size);
}

// -----------------------------------------------------------------------------------------------------------------------//

void coalesced_image_convolution(layer *pl, float *inputs) {
	int output_size = pl->k * pl->out_w * pl->out_h;
	int weight_size = pl->k*pl->c*pl->size*pl->size;
	size_t global[3] = { 13, 13, 1 };
	size_t local[3] ={ 13, 13, 1 };

	cl_kernel krnl_to_execute = clGetkrnl_coalesced_conv();

    clSetKernelArg(krnl_to_execute, 0, sizeof(cl_mem), &pl->mo_inputs);
    clSetKernelArg(krnl_to_execute, 1, sizeof(cl_mem), &pl->mo_weights);
    clSetKernelArg(krnl_to_execute, 2, sizeof(cl_mem), &pl->mo_outputs);
    clSetKernelArg(krnl_to_execute, 3, sizeof(int), &pl->k);
    clSetKernelArg(krnl_to_execute, 4, sizeof(int), &pl->c);
    clSetKernelArg(krnl_to_execute, 5, sizeof(int), &pl->w);
    clSetKernelArg(krnl_to_execute, 6, sizeof(int), &pl->h);
    clSetKernelArg(krnl_to_execute, 7, sizeof(int), &pl->size);
    clSetKernelArg(krnl_to_execute, 8, sizeof(int), &pl->out_w);
    clSetKernelArg(krnl_to_execute, 9, sizeof(int), &pl->out_h);
    clSetKernelArg(krnl_to_execute, 10, sizeof(int), &pl->pad);
    clSetKernelArg(krnl_to_execute, 11, sizeof(int), &pl->stride);
    clSetKernelArg(krnl_to_execute, 12, weight_size*sizeof(float), NULL);

    cl_run_kernel3d(krnl_to_execute, global, local, 3);

	cl_memcpy_from_device(pl->outputs, pl->mo_outputs, sizeof(float) * output_size);
}

// -----------------------------------------------------------------------------------------------------------------------//

void wgnum_image_convolution(layer *pl, float *inputs) {
	int output_size = pl->k * pl->out_w * pl->out_h;
	int weight_size = pl->k*pl->c*pl->size*pl->size;
	size_t global[3] = { 169, 169, 1 };
	size_t local[3] ={ 13, 13, 1 };

	cl_kernel krnl_to_execute = clGetkrnl_wgnum_conv();

    clSetKernelArg(krnl_to_execute, 0, sizeof(cl_mem), &pl->mo_inputs);
    clSetKernelArg(krnl_to_execute, 1, sizeof(cl_mem), &pl->mo_weights);
    clSetKernelArg(krnl_to_execute, 2, sizeof(cl_mem), &pl->mo_outputs);
    clSetKernelArg(krnl_to_execute, 3, sizeof(int), &pl->k);
    clSetKernelArg(krnl_to_execute, 4, sizeof(int), &pl->c);
    clSetKernelArg(krnl_to_execute, 5, sizeof(int), &pl->w);
    clSetKernelArg(krnl_to_execute, 6, sizeof(int), &pl->h);
    clSetKernelArg(krnl_to_execute, 7, sizeof(int), &pl->size);
    clSetKernelArg(krnl_to_execute, 8, sizeof(int), &pl->out_w);
    clSetKernelArg(krnl_to_execute, 9, sizeof(int), &pl->out_h);
    clSetKernelArg(krnl_to_execute, 10, sizeof(int), &pl->pad);
    clSetKernelArg(krnl_to_execute, 11, sizeof(int), &pl->stride);
    clSetKernelArg(krnl_to_execute, 12, weight_size*sizeof(float), NULL);

    cl_run_kernel3d(krnl_to_execute, global, local, 3);

	cl_memcpy_from_device(pl->outputs, pl->mo_outputs, sizeof(float) * output_size);
}

// -----------------------------------------------------------------------------------------------------------------------//

void wgnum_v2_image_convolution(layer *pl, float *inputs) {
	int output_size = pl->k * pl->out_w * pl->out_h;
	int weight_size = pl->k*pl->c*pl->size*pl->size;
	size_t global[3] = { 169, 169, 32 };
	size_t local[3] ={ 13, 13, 1 };

	cl_kernel krnl_to_execute = clGetkrnl_wgnum_v2_conv();

    clSetKernelArg(krnl_to_execute, 0, sizeof(cl_mem), &pl->mo_inputs);
    clSetKernelArg(krnl_to_execute, 1, sizeof(cl_mem), &pl->mo_weights);
    clSetKernelArg(krnl_to_execute, 2, sizeof(cl_mem), &pl->mo_outputs);
    clSetKernelArg(krnl_to_execute, 3, sizeof(int), &pl->k);
    clSetKernelArg(krnl_to_execute, 4, sizeof(int), &pl->c);
    clSetKernelArg(krnl_to_execute, 5, sizeof(int), &pl->w);
    clSetKernelArg(krnl_to_execute, 6, sizeof(int), &pl->h);
    clSetKernelArg(krnl_to_execute, 7, sizeof(int), &pl->size);
    clSetKernelArg(krnl_to_execute, 8, sizeof(int), &pl->out_w);
    clSetKernelArg(krnl_to_execute, 9, sizeof(int), &pl->out_h);
    clSetKernelArg(krnl_to_execute, 10, sizeof(int), &pl->pad);
    clSetKernelArg(krnl_to_execute, 11, sizeof(int), &pl->stride);
    clSetKernelArg(krnl_to_execute, 12, weight_size*sizeof(float), NULL);

    cl_run_kernel3d(krnl_to_execute, global, local, 3);

	cl_memcpy_from_device(pl->outputs, pl->mo_outputs, sizeof(float) * output_size);
}

// -----------------------------------------------------------------------------------------------------------------------//

void workload_image_convolution(layer *pl, float *inputs) {
	int output_size = pl->k * pl->out_w * pl->out_h;
	// int weight_size = pl->k*pl->c*pl->size*pl->size;
	size_t global[3] = { 169, 169, 32/pl->workload };
	size_t local[3] ={ 13, 13, 1 };

	cl_kernel krnl_to_execute = clGetkrnl_workload_conv(pl->workload);

    clSetKernelArg(krnl_to_execute, 0, sizeof(cl_mem), &pl->mo_pad_inputs);
    clSetKernelArg(krnl_to_execute, 1, sizeof(cl_mem), &pl->mo_weights);
    clSetKernelArg(krnl_to_execute, 2, sizeof(cl_mem), &pl->mo_outputs);
    clSetKernelArg(krnl_to_execute, 3, sizeof(int), &pl->k);
    clSetKernelArg(krnl_to_execute, 4, sizeof(int), &pl->c);
    clSetKernelArg(krnl_to_execute, 5, sizeof(int), &pl->w);
    clSetKernelArg(krnl_to_execute, 6, sizeof(int), &pl->h);
    clSetKernelArg(krnl_to_execute, 7, sizeof(int), &pl->size);
    clSetKernelArg(krnl_to_execute, 8, sizeof(int), &pl->out_w);
    clSetKernelArg(krnl_to_execute, 9, sizeof(int), &pl->out_h);
    clSetKernelArg(krnl_to_execute, 10, sizeof(int), &pl->pad);
    clSetKernelArg(krnl_to_execute, 11, sizeof(int), &pl->stride);
    // clSetKernelArg(krnl_to_execute, 11, sizeof(int), &pl->workload);
    // clSetKernelArg(krnl_to_execute, 12, weight_size*sizeof(float), NULL);

    cl_run_kernel3d(krnl_to_execute, global, local, 3);

	cl_memcpy_from_device(pl->outputs, pl->mo_outputs, sizeof(float) * output_size);
}

// -----------------------------------------------------------------------------------------------------------------------//

void fixpoint_image_convolution(layer *pl, float *inputs) {
	int output_size = pl->k * pl->out_w * pl->out_h;
	int weight_size = pl->k*pl->c*pl->size*pl->size;
	size_t global[3] = { 13, 13, 1 };
	size_t local[3] ={ 13, 13, 1 };

	cl_kernel krnl_to_execute = clGetkrnl_fixpoint_conv(pl->workload);

    clSetKernelArg(krnl_to_execute, 0, sizeof(cl_mem), &pl->mo_inputs);
    clSetKernelArg(krnl_to_execute, 1, sizeof(cl_mem), &pl->mo_weights);
    clSetKernelArg(krnl_to_execute, 2, sizeof(cl_mem), &pl->mo_outputs);
    clSetKernelArg(krnl_to_execute, 3, sizeof(int), &pl->k);
    clSetKernelArg(krnl_to_execute, 4, sizeof(int), &pl->c);
    clSetKernelArg(krnl_to_execute, 5, sizeof(int), &pl->w);
    clSetKernelArg(krnl_to_execute, 6, sizeof(int), &pl->h);
    clSetKernelArg(krnl_to_execute, 7, sizeof(int), &pl->size);
    clSetKernelArg(krnl_to_execute, 8, sizeof(int), &pl->out_w);
    clSetKernelArg(krnl_to_execute, 9, sizeof(int), &pl->out_h);
    clSetKernelArg(krnl_to_execute, 10, sizeof(int), &pl->pad);
    clSetKernelArg(krnl_to_execute, 11, sizeof(int), &pl->stride);
    clSetKernelArg(krnl_to_execute, 12, weight_size*sizeof(float), NULL);

    cl_run_kernel3d(krnl_to_execute, global, local, 3);

	cl_memcpy_from_device(pl->outputs, pl->mo_outputs, sizeof(float) * output_size);
}

// -----------------------------------------------------------------------------------------------------------------------//

void gpu_image_convolution(layer *pl, float *inputs, int flag) {
	switch(flag) {
		case 1: original_image_convolution	(pl, inputs); break;
		case 2: local_image_convolution		(pl, inputs); break;
		case 3: memorize_image_convolution	(pl, inputs); break;
		case 4: wgsize_image_convolution	(pl, inputs); break;
		case 5: coalesced_image_convolution	(pl, inputs); break;
		case 6: wgnum_image_convolution		(pl, inputs); break;
		case 7: wgnum_v2_image_convolution	(pl, inputs); break;
		case 8: workload_image_convolution	(pl, inputs); break;
		case 9: fixpoint_image_convolution	(pl, inputs); break;
		default: ;
	}
}
#endif // OPENCL