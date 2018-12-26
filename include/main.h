#ifndef MAIN_API
#define MAIN_API
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "define_cl.h"

typedef struct layer{
	int k;
	int c;
	int w;
	int h;
	int out_w;
	int out_h;
	int stride;
	int pad;
	int size;
	float *weights;
	float *outputs;
#ifdef OPENCL
	int workload;
	cl_mem mo_pad_inputs;
	cl_mem mo_fixed_weights;

	cl_mem mo_inputs;
	cl_mem mo_weights;
	cl_mem mo_outputs;
#endif //OPENCL
}layer;


#endif // MAIN_API