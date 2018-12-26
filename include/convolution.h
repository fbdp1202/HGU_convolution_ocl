#ifndef __CONV_H
#define __CONV_H

#include "layer.h"

void test_convolution();

#ifdef OPENCL
void init_ocl_convolution(layer *pl, float *inputs);
void free_ocl(layer *pl);
void gpu_image_convolution(layer *pl, float *inputs, int flag);
#endif // OPENCL

#endif // __CONV_H