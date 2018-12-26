#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <dirent.h>

#include "layer.h"
#include "image.h"
#include "utils.h"
#include "convolution.h"
#include "define_cl.h"

void init_convolution_size(layer *pl) {
	printf("Output Channel: ");
	scanf(" %d", &pl->k);

	printf("Input Channel: ");
	scanf(" %d", &pl->c);

	printf("Input width and height: ");
	scanf(" %d", &pl->w);
	pl->h = pl->w;

	printf("Weight filter width and height: ");
	scanf(" %d", &pl->size);

	pl->stride = 1;
	pl->pad = 1;
	pl->out_h = pl->out_w = (pl->w + 2*pl->pad - pl->size + 1)/pl->stride;
	puts("");

}

void load_convolution_config(layer *pl) {
	printf("LOAD config in layer.txt\n\n");
	FILE *fp = fopen("layer.txt", "r");
	if (fp == NULL) {
		printf("layer.txt can't open\n");
		exit(1);
	}
	fscanf(fp, " %d", &pl->k);
	fscanf(fp, " %d", &pl->c);
	fscanf(fp, " %d", &pl->w);
	pl->h = pl->w;
	fscanf(fp, " %d", &pl->size);
	pl->stride = 1;
	pl->pad = 1;
	pl->out_h = pl->out_w = (pl->w + 2*pl->pad - pl->size + 1)/pl->stride;
	fclose(fp);
}

void print_convolution_size(layer *pl) {
	printf("Output Channel: %d\n", pl->k);
	printf("Input Channel: %d\n", pl->c);
	printf("Input width: %d\n", pl->w);
	printf("Input height: %d\n", pl->h);
	printf("stride height: %d\n", pl->stride);
	printf("pad height: %d\n", pl->pad);
	printf("Weight filter width: %d\n", pl->size);
	printf("Weight filter height: %d\n", pl->size);
	printf("Output width: %d\n", pl->out_w);
	printf("Output height: %d\n", pl->out_h);

	puts("");
}

void init_convolution_weight(layer *pl) {
	int i;
	printf("SET RAMDOM WEIGHT\n");
	int nweight = pl->k * pl->c * pl->size * pl->size;
	pl->weights = (float *)calloc(nweight, sizeof(float));
	for (i=0; i<nweight; i++)
		pl->weights[i] = gaussianRandom();
}

void init_convolution_output(layer *pl) {
	int noutput = pl->k * pl->out_w * pl->out_h;
	pl->outputs = (float *)calloc(noutput, sizeof(float));
}

layer init_convolution() {
	layer l;
	// init_convolution_size(&l);
	load_convolution_config(&l);
	print_convolution_size(&l);
	init_convolution_weight(&l);
	init_convolution_output(&l);
	return l;
}

void cpu_image_convolution(layer *pl, float *inputs) {
	int k, c, w, h, mh, mw;
	int outpus_size = pl->out_w * pl->out_h;
	for (k=0; k<pl->k; k++) {
		for (h=0; h<pl->out_h; h++) {
			for (w=0; w<pl->out_w; w++) {
				float ret = 0;
				for (c=0; c<pl->c; c++) {
					for (mh=0; mh<pl->size; mh++) {
						int curr_h = h * pl->stride + mh - pl->pad;
						if (curr_h < 0 || curr_h >= pl->h) continue;
						for (mw=0; mw<pl->size; mw++) {
							int curr_w = w * pl->stride + mw - pl->pad;
							if (curr_w < 0 || curr_w >= pl->w) continue;
							int inputIdx = (c*pl->h+curr_h)*pl->w+curr_w;
							int weightIdx = ((k*pl->c+c)*pl->size+mh)*pl->size+mw;
							ret += inputs[inputIdx] * pl->weights[weightIdx];
						}
					}
				}
				pl->outputs[k*outpus_size + h*pl->out_w + w] = ret;
			}
		}
	}
}

void cal_convolution(layer *pl, float *inputs, int flag) {
	int i;
	int num_cycle = 0;
	while (num_cycle<=0) {
		printf("Number of repeat: ");
		scanf(" %d", &num_cycle);
	}
	printf("CPU image convolution\n");
	double start_cpu_time = what_time_is_it_now();
	for (i=0; i<num_cycle; i++) {
		cpu_image_convolution(pl, inputs);
	}
	printf("CPU imge convolution Time: %f seconds\n", (what_time_is_it_now()-start_cpu_time)/num_cycle);
#ifdef OPENCL
	printf("GPU image convolution\n");
	init_ocl_convolution(pl, inputs);

	double start_gpu_time = what_time_is_it_now();
	for (i=0; i<num_cycle; i++) {
		gpu_image_convolution(pl, inputs, flag);
	}
	printf("GPU image convolution Time: %f seconds\n", (what_time_is_it_now()-start_gpu_time)/num_cycle);
#endif //OPENCL
}

void delete_convolution(layer *pl) {
	if (pl->weights) free(pl->weights);
	if (pl->outputs) free(pl->outputs);
}

void test_convolution() {
	int choose = 0;
#ifdef OPENCL
	clSetup();
	printf("**** Choose mode ****\n");
	printf("1) Original_convolution\n");
	printf("2) Local_convolution\n");
	printf("3) memorize_convolution\n");
	printf("4) Work Group size\t\t[  1 x  13 x  13, 1 x 13 x 13 ]\n");
	printf("5) Coalesced memory\t\t[  1 x  13 x  13, 1 x 13 x 13 ]\n");
	printf("6) Work Group num\t\t[  1 x 169 x 169, 1 x 13 x 13 ]\n");
	printf("7) Work Group_v2 num\t\t[ 32 x 169 x 169, 1 x 13 x 13 ]\n");
	printf("8) Work Load\t\t\t[ 16 x 169 x 169, {2,4,8,16} x 13 x 13 ]\n");
	printf("9) Fixed Point weight\n");

	scanf(" %d", &choose);
	switch(choose) {
		case 1: clKernelSetup("./ocl/original_conv.cl"); break;
		case 2: clKernelSetup("./ocl/local_conv.cl"); break;
		case 3: clKernelSetup("./ocl/memorize_conv.cl"); break;
		case 4: clKernelSetup("./ocl/wgsize_conv.cl"); break;
		case 5: clKernelSetup("./ocl/coalesced_conv.cl"); break;
		case 6: clKernelSetup("./ocl/wgnum_conv.cl"); break;
		case 7: clKernelSetup("./ocl/wgnum_v2_conv.cl"); break;
		case 8: clKernelSetup("./ocl/workload_conv.cl"); break;
		case 9: clKernelSetup("./ocl/fixpoint_conv.cl"); break;
		default:;
	}
#endif // OPENCL
	layer l = init_convolution();

	// searchInDirectory(".");
	// puts("");

	// DIR *d;
	// struct dirent *dir;
	// d = opendir(".ocl/");
	// if (d)
	// {
	// 	while ( (dir = readdir(d)) != NULL) {
	// 		printf("%s\n", dir->d_name);
	// 	}
	// 	closedir(d);
	// }
	// else printf("cannot open directory\n");

	int input_size = l.c*l.w*l.h;
	unsigned char *inputs = calloc(input_size, sizeof(unsigned char));
	set_random_input(inputs, input_size);

	float *X = calloc(input_size, sizeof(float));
	copy_input(inputs, X, input_size);
	cal_convolution(&l, X, choose);

	free(X);
	free(inputs);
	delete_convolution(&l);
}