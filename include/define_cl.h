#ifndef DEFINE_CL_H
#define DEFINE_CL_H

#ifdef OPENCL
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>

#define VERSION_GAMMA
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LOCAL_DEPTH 8
#define MAX_SHORT ((int)(pow(2.0, (double)(l.wbitlen))) - 1)
#define MAX_FIXED 255

#define CL_ERR_TO_STR(err) case err: return #err
#define CL_LOG_SIZE 8196
char const* clGetErrorString(cl_int const err);

cl_kernel clGetkrnl_orig_conv();
cl_kernel clGetkrnl_local_conv();
cl_kernel clGetkrnl_memorize_conv();
cl_kernel clGetkrnl_wgsize_conv();
cl_kernel clGetkrnl_coalesced_conv();
cl_kernel clGetkrnl_wgnum_conv();
cl_kernel clGetkrnl_wgnum_v2_conv();
cl_kernel clGetkrnl_workload_conv();
cl_kernel clGetkrnl_fixpoint_conv();

void clSetup();
void clKernelSetup(const char *krnl_file);
void clReleaseAll();

void clSetKrnlArg(cl_kernel krnl, cl_uint num, size_t size, void *ptr);

cl_mem clCreateMemobj(cl_mem_flags flags, size_t size, float* host_ptr);
void clFreeMemobj(cl_mem buffer);

void cl_memcpy_to_device(cl_mem dest, void* src, size_t size);
void cl_memcpy_from_device(void* dest, cl_mem src, size_t size);
void cl_run_kernel3d(cl_kernel krnl, size_t* global, size_t* local, cl_uint workDim);
void cl_run_kernel3d_async(cl_kernel krnl, size_t* global, size_t* local, cl_uint workDim);
void cl_estimate_time(cl_event event, const char*);

void define_log(const char* comment);

char log_sw;

float poclu_cl_half_to_float(cl_half value);
cl_half poclu_float_to_cl_half(float value); 

#endif

#endif