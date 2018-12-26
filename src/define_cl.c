#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "define_cl.h"
#ifdef OPENCL

cl_platform_id platform_id = NULL;
cl_device_id device_id = NULL;
cl_context context = NULL;
cl_program program = NULL;
cl_command_queue command_queue = NULL;

cl_kernel krnl_orig_conv = NULL;
cl_kernel krnl_local_conv = NULL;
cl_kernel krnl_memorize_conv = NULL;
cl_kernel krnl_wgsize_conv = NULL;
cl_kernel krnl_coalesced_conv = NULL;
cl_kernel krnl_wgnum_conv = NULL;
cl_kernel krnl_wgnum_v2_conv = NULL;
cl_kernel krnl_workload_conv[33] = {NULL,};
cl_kernel krnl_fixpoint_conv[33] = {NULL,};

// cl_mem clGetMem_d_a() { return d_a; }
// cl_mem* clGetpMem_d_a() { return &d_a; }

cl_kernel clGetkrnl_orig_conv() { return krnl_orig_conv; }
cl_kernel clGetkrnl_local_conv() { return krnl_local_conv; }
cl_kernel clGetkrnl_memorize_conv() { return krnl_memorize_conv; }
cl_kernel clGetkrnl_wgsize_conv() { return krnl_wgsize_conv; }
cl_kernel clGetkrnl_coalesced_conv() { return krnl_coalesced_conv; }
cl_kernel clGetkrnl_wgnum_conv() { return krnl_wgnum_v2_conv; }
cl_kernel clGetkrnl_wgnum_v2_conv() { return krnl_wgnum_v2_conv; }
cl_kernel clGetkrnl_workload_conv(int workload) { return krnl_workload_conv[workload]; }
cl_kernel clGetkrnl_fixpoint_conv(int workload) { return krnl_fixpoint_conv[workload]; }



void Deivce_info(cl_platform_id platform, cl_device_id device){
	cl_int err;
	cl_char string[10240] = {0};

    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(string), &string, NULL);
    printf("Platform: %s\n", string);

    err = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(string), &string, NULL);
    printf("Vendor: %s\n", string);

    err = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(string), &string, NULL);
    printf("Version: %s\n", string);

    printf("\t-------------------------\n");

    // Get device name
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(string), &string, NULL);
    printf("\t\tName: %s\n", string);

    // Get device OpenCL version
    err = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(string), &string, NULL);
    printf("\t\tVersion: %s\n", string);

    // Get Max. Compute units
    cl_uint num;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num, NULL);
    printf("\t\tMax. Compute Units: %d\n", num);

    // Get local memory size
    cl_ulong mem_size;
    err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL);
    printf("\t\tLocal Memory Size: %llu KB\n", mem_size/1024);

    // Get global memory size
    err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL);
    printf("\t\tGlobal Memory Size: %llu MB\n", mem_size/(1024*1024));

    // Get maximum buffer alloc. size
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &mem_size, NULL);
    printf("\t\tMax Alloc Size: %llu MB\n", mem_size/(1024*1024));

    // Get work-group size information
    size_t size;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &size, NULL);
    printf("\t\tMax Work-group Total Size: %ld\n", size);

    // Find the maximum dimensions of the work-groups
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &num, NULL);
    // Get the max. dimensions of the work-groups
    size_t dims[num];
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(dims), &dims, NULL);
    printf("\t\tMax Work-group Dims: ( ");
    size_t k;
    for (k = 0; k < num; k++)
    {
        printf("%ld ", dims[k]);
    }
    printf(")\n");

    printf("\t-------------------------\n");

}

char * getKernelSource(const char *filename)
{
	FILE *file = fopen(filename, "r");
	if (!file)
	{
		fprintf(stderr, "Error: Could not open kernel source file\n");
		exit(EXIT_FAILURE);
	}
	fseek(file, 0, SEEK_END);
	int len = ftell(file) + 1;
	rewind(file);

	char *source = (char *)calloc(sizeof(char), len);
	if (!source)
	{
		fprintf(stderr, "Error: Could not allocate memory for source string\n");
		exit(EXIT_FAILURE);
	}
	fread(source, sizeof(char), len, file);
	fclose(file);
	return source;
}

char const* clGetErrorString(cl_int const err) {
	switch (err)
	{
		CL_ERR_TO_STR(CL_SUCCESS);
		CL_ERR_TO_STR(CL_DEVICE_NOT_FOUND);
		CL_ERR_TO_STR(CL_DEVICE_NOT_AVAILABLE);
		CL_ERR_TO_STR(CL_COMPILER_NOT_AVAILABLE);
		CL_ERR_TO_STR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
		CL_ERR_TO_STR(CL_OUT_OF_RESOURCES);
		CL_ERR_TO_STR(CL_OUT_OF_HOST_MEMORY);
		CL_ERR_TO_STR(CL_PROFILING_INFO_NOT_AVAILABLE);
		CL_ERR_TO_STR(CL_MEM_COPY_OVERLAP);
		CL_ERR_TO_STR(CL_IMAGE_FORMAT_MISMATCH);
		CL_ERR_TO_STR(CL_IMAGE_FORMAT_NOT_SUPPORTED);
		CL_ERR_TO_STR(CL_BUILD_PROGRAM_FAILURE);
		CL_ERR_TO_STR(CL_MAP_FAILURE);
		CL_ERR_TO_STR(CL_MISALIGNED_SUB_BUFFER_OFFSET);
		CL_ERR_TO_STR(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
		CL_ERR_TO_STR(CL_COMPILE_PROGRAM_FAILURE);
		CL_ERR_TO_STR(CL_LINKER_NOT_AVAILABLE);
		CL_ERR_TO_STR(CL_LINK_PROGRAM_FAILURE);
		CL_ERR_TO_STR(CL_DEVICE_PARTITION_FAILED);
		CL_ERR_TO_STR(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
		CL_ERR_TO_STR(CL_INVALID_VALUE);
		CL_ERR_TO_STR(CL_INVALID_DEVICE_TYPE);
		CL_ERR_TO_STR(CL_INVALID_PLATFORM);
		CL_ERR_TO_STR(CL_INVALID_DEVICE);
		CL_ERR_TO_STR(CL_INVALID_CONTEXT);
		CL_ERR_TO_STR(CL_INVALID_QUEUE_PROPERTIES);
		CL_ERR_TO_STR(CL_INVALID_COMMAND_QUEUE);
		CL_ERR_TO_STR(CL_INVALID_HOST_PTR);
		CL_ERR_TO_STR(CL_INVALID_MEM_OBJECT);
		CL_ERR_TO_STR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
		CL_ERR_TO_STR(CL_INVALID_IMAGE_SIZE);
		CL_ERR_TO_STR(CL_INVALID_SAMPLER);
		CL_ERR_TO_STR(CL_INVALID_BINARY);
		CL_ERR_TO_STR(CL_INVALID_BUILD_OPTIONS);
		CL_ERR_TO_STR(CL_INVALID_PROGRAM);
		CL_ERR_TO_STR(CL_INVALID_PROGRAM_EXECUTABLE);
		CL_ERR_TO_STR(CL_INVALID_KERNEL_NAME);
		CL_ERR_TO_STR(CL_INVALID_KERNEL_DEFINITION);
		CL_ERR_TO_STR(CL_INVALID_KERNEL);
		CL_ERR_TO_STR(CL_INVALID_ARG_INDEX);
		CL_ERR_TO_STR(CL_INVALID_ARG_VALUE);
		CL_ERR_TO_STR(CL_INVALID_ARG_SIZE);
		CL_ERR_TO_STR(CL_INVALID_KERNEL_ARGS);
		CL_ERR_TO_STR(CL_INVALID_WORK_DIMENSION);
		CL_ERR_TO_STR(CL_INVALID_WORK_GROUP_SIZE);
		CL_ERR_TO_STR(CL_INVALID_WORK_ITEM_SIZE);
		CL_ERR_TO_STR(CL_INVALID_GLOBAL_OFFSET);
		CL_ERR_TO_STR(CL_INVALID_EVENT_WAIT_LIST);
		CL_ERR_TO_STR(CL_INVALID_EVENT);
		CL_ERR_TO_STR(CL_INVALID_OPERATION);
		CL_ERR_TO_STR(CL_INVALID_GL_OBJECT);
		CL_ERR_TO_STR(CL_INVALID_BUFFER_SIZE);
		CL_ERR_TO_STR(CL_INVALID_MIP_LEVEL);
		CL_ERR_TO_STR(CL_INVALID_GLOBAL_WORK_SIZE);
		CL_ERR_TO_STR(CL_INVALID_PROPERTY);
		CL_ERR_TO_STR(CL_INVALID_IMAGE_DESCRIPTOR);
		CL_ERR_TO_STR(CL_INVALID_COMPILER_OPTIONS);
		CL_ERR_TO_STR(CL_INVALID_LINKER_OPTIONS);
		CL_ERR_TO_STR(CL_INVALID_DEVICE_PARTITION_COUNT);
		//CL_ERR_TO_STR(CL_INVALID_PIPE_SIZE);
		//CL_ERR_TO_STR(CL_INVALID_DEVICE_QUEUE);

	default:
		return "UNKNOWN ERROR CODE";
	}
}

void clSetup() {
	cl_int err;
	char log[CL_LOG_SIZE] = { 0 };
	size_t log_size;

	//platform
	cl_uint ret_num_platforms;
	err = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	if (err != CL_SUCCESS) {
		printf("Error: no platforms available or OpenCL install broken");
		exit(EXIT_FAILURE);
	}

	//device
	cl_uint ret_num_devices;
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to get the number of devices: %s\n", clGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	Deivce_info(platform_id, device_id);

	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to create a compute context! : %s\n", clGetErrorString(err));
		printf("Test failed\n");
		exit(EXIT_FAILURE);
	}

	command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to create a command queue! : %s\n", clGetErrorString(err));
		printf("Test failed\n");
		exit(EXIT_FAILURE);
	}

	printf("create command_queue\n");
}

void clKernelSetup(const char *krnl_file)
{
	cl_int err;	
	char log[CL_LOG_SIZE] = { 0 };
	size_t log_size;
	char *krnl_bin;
	krnl_bin = getKernelSource(krnl_file);

	program = clCreateProgramWithSource(context, 1, (const char **)&krnl_bin, NULL, &err);
	if ((!program) || (err != CL_SUCCESS)) {
		printf("Error: Failed to create compute program %s\n", clGetErrorString(err));
		printf("Test failed\n");
		exit(EXIT_FAILURE);
	}
	printf("create Program\n");

	err = clBuildProgram(program, 1, &device_id, "", NULL, NULL);
	//err = clBuildProgram(program, 1, &device_id, "-cl-nv-maxrregcount=1024", NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to build program executable! : %s\n", clGetErrorString(err));

		err = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(log), log, &log_size);
		if (err != CL_SUCCESS)
			printf("Error: Failed to load err log string! : %s\n", clGetErrorString(err));
		else
			printf("%s", log);

		exit(EXIT_FAILURE);
	}
	printf("clBuildProgram\n");

// -----------------------------------------------------------------------------------------------------------------------//
	if (!strcmp(krnl_file, "./ocl/original_conv.cl")) {
		krnl_orig_conv = clCreateKernel(program, "image_convolution", &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create kernel for image_convolution: %s\n", clGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	}
// -----------------------------------------------------------------------------------------------------------------------//
	if (!strcmp(krnl_file, "./ocl/local_conv.cl")) {
		krnl_local_conv = clCreateKernel(program, "local_convolution", &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create kernel for local_convolution: %s\n", clGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	}
// -----------------------------------------------------------------------------------------------------------------------//
	if (!strcmp(krnl_file, "./ocl/memorize_conv.cl")) {
		krnl_memorize_conv = clCreateKernel(program, "memorize_convolution", &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create kernel for memorize_convolution: %s\n", clGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	}
// -----------------------------------------------------------------------------------------------------------------------//
	if (!strcmp(krnl_file, "./ocl/wgsize_conv.cl")) {
		krnl_wgsize_conv = clCreateKernel(program, "wgsize_convolution", &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create kernel for wgsize_convolution: %s\n", clGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	}
// -----------------------------------------------------------------------------------------------------------------------//
	if (!strcmp(krnl_file, "./ocl/coalesced_conv.cl")) {
		krnl_coalesced_conv = clCreateKernel(program, "coalesced_convolution", &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create kernel for coalesced_convolution: %s\n", clGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	}
// -----------------------------------------------------------------------------------------------------------------------//
	if (!strcmp(krnl_file, "./ocl/wgnum_conv.cl")) {
		krnl_wgnum_conv = clCreateKernel(program, "wgnum_convolution", &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create kernel for coalesced_convolution: %s\n", clGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	}
// -----------------------------------------------------------------------------------------------------------------------//
	if (!strcmp(krnl_file, "./ocl/wgnum_v2_conv.cl")) {
		krnl_wgnum_v2_conv = clCreateKernel(program, "wgnum_v2_convolution", &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create kernel for coalesced_convolution: %s\n", clGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	}
// -----------------------------------------------------------------------------------------------------------------------//
	if (!strcmp(krnl_file, "./ocl/workload_conv.cl")) {
		krnl_workload_conv[1] = clCreateKernel(program, "workload1_convolution", &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create kernel for workload1_convolution: %s\n", clGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		krnl_workload_conv[2] = clCreateKernel(program, "workload2_convolution", &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create kernel for workload2_convolution: %s\n", clGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		krnl_workload_conv[4] = clCreateKernel(program, "workload4_convolution", &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create kernel for workload4_convolution: %s\n", clGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		krnl_workload_conv[8] = clCreateKernel(program, "workload8_convolution", &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create kernel for workload8_convolution: %s\n", clGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		krnl_workload_conv[16] = clCreateKernel(program, "workload16_convolution", &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create kernel for workload16_convolution: %s\n", clGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		krnl_workload_conv[32] = clCreateKernel(program, "workload32_convolution", &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create kernel for workload32_convolution: %s\n", clGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	}
// -----------------------------------------------------------------------------------------------------------------------//
	if (!strcmp(krnl_file, "./ocl/fixpoint_conv.cl")) {
		krnl_fixpoint_conv[2] = clCreateKernel(program, "fixpoint2_convolution", &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create kernel for coalesced_convolution: %s\n", clGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		krnl_fixpoint_conv[4] = clCreateKernel(program, "fixpoint4_convolution", &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create kernel for coalesced_convolution: %s\n", clGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		krnl_fixpoint_conv[8] = clCreateKernel(program, "fixpoint8_convolution", &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create kernel for coalesced_convolution: %s\n", clGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		krnl_fixpoint_conv[16] = clCreateKernel(program, "fixpoint16_convolution", &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create kernel for coalesced_convolution: %s\n", clGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		krnl_fixpoint_conv[32] = clCreateKernel(program, "fixpoint32_convolution", &err);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to create kernel for coalesced_convolution: %s\n", clGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	}
// -----------------------------------------------------------------------------------------------------------------------//
	// size_t local;

	// err = clGetKernelWorkGroupInfo(krnl_orig_conv, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
	// if (err != CL_SUCCESS)
	// {
	// 	printf("Error: Failed to retrieve kernel work group info! %d\n", err);
	// 	exit(1);
	// }
 //    printf("\t\tCL_KERNEL_WORK_GROUP_SIZE Total Size: %ld\n", local);

	// err = clGetKernelWorkGroupInfo(krnl_orig_conv, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(local), &local, NULL);
	// if (err != CL_SUCCESS)
	// {
	// 	printf("Error: Failed to retrieve CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE info! %d\n", err);
	// 	exit(1);
	// }
 //    printf("\t\tCL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE Total Size: %ld\n", local);

	clReleaseProgram(program);
}

void clReleaseAll() {
	clFlush(command_queue);
	clFinish(command_queue);
	
	clReleaseKernel(krnl_orig_conv);

	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
}

void clSetKrnlArg(cl_kernel krnl, cl_uint num, size_t size, void *ptr) {
	int err = clSetKernelArg(krnl, num, size, ptr);

	if (err != CL_SUCCESS) {
		printf("Error: Failed to set kernel arg %s\n", clGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

cl_mem clCreateMemobj(cl_mem_flags flags, size_t size, float* host_ptr) {
	int errNum = 0;
	cl_mem mem = clCreateBuffer(context, flags, size, host_ptr, &errNum);

	if (!mem || errNum != CL_SUCCESS) {
		printf("Error: Failed to allocate device memory!: %s\n", clGetErrorString(errNum));
		exit(EXIT_FAILURE);
	}

	return mem;
}

void clFreeMemobj(cl_mem buffer) {
	clReleaseMemObject(buffer);
}

void cl_memcpy_to_device(cl_mem dest, void* src,
	size_t size) {

	cl_event event;

	int err = clEnqueueWriteBuffer(command_queue, dest, CL_TRUE, 0, size,
		src, 0, NULL, &event);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to write to source array a!\n");
		exit(EXIT_FAILURE);
	}
	if(log_sw)
		cl_estimate_time(event, "memcpy to device Time");
}

void cl_memcpy_from_device(void* dest, cl_mem src,
	size_t size) {

	cl_event event;

	int err = clEnqueueReadBuffer(command_queue, src, CL_TRUE, 0, size,
		dest, 0, NULL, &event);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to read output array! %s\n", clGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	if(log_sw)
		cl_estimate_time(event, "memcpy from device Time");
}

void cl_run_kernel3d(cl_kernel krnl, size_t* global, size_t* local, cl_uint workDim) {
	cl_event event;

	int err = clEnqueueNDRangeKernel(command_queue, krnl, workDim,
		NULL, global, local, 0, NULL, &event);
	if (err != CL_SUCCESS) {
		printf("Error: failed to execute kernel! %s\n", clGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	clWaitForEvents(1, &event);

	if(log_sw)
		cl_estimate_time(event, "Kernel Time");

	clReleaseEvent(event);
}

void cl_run_kernel3d_async(cl_kernel krnl, size_t* global, size_t* local, cl_uint workDim) {

	int err = clEnqueueNDRangeKernel(command_queue, krnl, workDim,
		NULL, global, local, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error: failed to execute kernel! %s\n", clGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void cl_estimate_time(cl_event event, const char* comment){
	cl_ulong start, end;
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);	
	float time = (end-start)*1.0e-3f;
	if(time < 1e10){
		printf("%s:\t%.6f\n",comment,time);
	}
	else{
		printf("%s:\tUnknown\n", comment);
	}
}


#ifndef INFINITY
#define INFINITY 1.0/0.0
#endif

#ifndef NAN
#define NAN 0.0/0.0
#endif

typedef union 
{
  int32_t i;
  float f;
} FloatConvUnion;

cl_half poclu_float_to_cl_half(float value) 
{
  FloatConvUnion u;
  u.f = value;
  cl_half half = (u.i >> 16) & 0x8000; // sign
  cl_half fraction = (u.i >> 12) & 0x007ff; // fraction with extra bit for rounding
  cl_half exponent = (u.i >> 23)  & 0xff; // exponent
  
  if(exponent < 0x0067) // Return signed zero if zero or value is too small for denormal half
    return half;

  if(exponent > 0x008e){// value was NaN or Inf
    half |= 0x7c00u; // Make into inf
    half |= exponent == 255 && (u.i & 0x007fffffu); // If value was NaN make this into NaN
    return half;
  }

  if(exponent < 0x0071){// Denormal
    fraction |= 0x0800u;

    // rounding
    half |= (fraction >> (0x0072 - exponent)) + ((fraction >> (0x0071 - exponent)) & 1);
    return half;
  }

  half |= ((exponent - 0x0070) << 10) | (fraction >> 1);
  half += fraction & 1;// rounding
  return half;
}

float poclu_cl_half_to_float(cl_half value) 
{
  if (value == 0xFC00) {
    return -INFINITY;
  }
  if (value == 0x7C00) {
    return INFINITY;
  }

  int sgn = ((value & 0x8000) >> 15);
  int exp = (value & 0x7C00) >> 10;
  int mant = value & 0x03FF;

  if (exp == 0x1F && mant != 0) {
    return NAN;
  }

  float v = (exp == 0) ? mant : mant | 0x0400; // 1.x if not denormal
  v /= 0x400;
  float mul = exp2((float)exp - 15);
  v *= mul;
  if (sgn) {
    v *= -1;
  }
  return v;
}

void do_conversion_h_to_f(float *to, cl_half *from, int size){
    int i;
    for(i = 0; i<size; i++){
        to[i] = poclu_cl_half_to_float(from[i]);
    }
}

void do_conversion_f_to_h(cl_half *to, float *from, int size){
    int i;
    for(i=0; i<size; i++){
        to[i] = poclu_float_to_cl_half(from[i]);
    }
}


void do_conversion_f_to_s(short *to, float *from, int size)
{
	int i;
	for(i=0; i<size; i++) to[i] = from[i];
}

void do_conversion_s_to_f(float *to, short *from, int size)
{
	int i;
	for(i=0; i<size; i++) to[i] = from[i];
}

void display_value_range(float *target, int size, int index)
{
	int i;
	char buf[256];
	sprintf(buf, "weight_%d",index);
	FILE *fp = fopen(buf, "w");
	for(i=0; i<size; i++) fprintf(fp,"%.6f\n", target[i]);
	fclose(fp);
}

void calculate_noise(float *target, int size)
{
	int i, c;
	float noise[15] = {0,};
	int num = 1;
	float tmp;
	for(c = 0; c <15; c++){
		for(i = 0; i < size; i++){
			tmp = target[i] - (float)round(target[i]*num)/(float)num;
			if(tmp < 0) tmp = -tmp;
			noise[c] += tmp;
		}
		printf("noise[%d]: %f\n", c, noise[c]/(float)size);
		num *= 2;
	}
}

#endif