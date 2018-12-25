#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "layer.h"
#include "image.h"
#include "utils.h"
#include "convolution.h"

void init_convolution_size(layer *pl) {
#ifdef DEBUG_MODE
	LOG_FUN();
#endif
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

void print_convolution_size(layer *pl) {
#ifdef DEBUG_MODE
	LOG_FUN();
#endif
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
#ifdef DEBUG_MODE
	LOG_FUN();
#endif
	int i;
	printf("SET RAMDOM WEIGHT\n");
	int nweight = pl->k * pl->c * pl->size * pl->size;
	pl->weight = (float *)calloc(nweight, sizeof(float));
	for (i=0; i<nweight; i++)
		pl->weights[i] = gaussianRandom();
}

void init_convolution_output(layer *pl) {
#ifdef DEBUG_MODE
	LOG_FUN();
#endif
	int noutput = pl->k * pl->out_w * pl->out_h;
	pl->outputs = (float *)calloc(noutput, sizeof(float));
}

layer init_convolution() {
#ifdef DEBUG_MODE
	LOG_FUN();
#endif
	layer l;
	init_convolution_size(&l);
	print_convolution_size(&l);
	init_convolution_weight(&l);
	init_convolution_output(&l);
	return l;
}

void delete_convolution(layer *pl) {
#ifdef DEBUG_MODE
	LOG_FUN();
#endif
	if (pl->weights) free(pl->weights);
	if (pl->outputs) free(pl->outputs);
}

void test_convolution() {
#ifdef DEBUG_MODE
	LOG_FUN();
#endif
	layer l = init_convolution();
	int input_size = l.c*l.w*l.h;
	unsigned char* inputs = make_random_input(input_size);
	fiubntout
	cal_convolution();
	delete_convolution();
}