#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "layer.h"
#include "convolution.h"

void init_convolution_size(layer *pl) {
	printf("Output Channel: ");
	scanf(" %d", &pl->k);

	printf("Input Channel: ");
	scanf(" %d", &pl->c);

	printf("input width: ");
	scanf(" %d", &pl->w);

	printf("input height: ");
	scanf(" %d", &pl->h);
}

void print_convolution_size(layer *pl) {
	printf("Output Channel: ");
	printf(" %d\n", pl->k);

	printf("Input Channel: ");
	printf(" %d\n", pl->c);

	printf("input width: ");
	printf(" %d\n", pl->w);

	printf("input height: ");
	printf(" %d\n", pl->h);
}

layer init_convolution() {
	printf("init_convolution\n");
	layer l;
	init_convolution_size(&l);
	printf("Check Config Value\n");
	print_convolution_size(&l);
	return l;
}

void test_convolution() {
	printf("start test_convolution!\n");
	layer l = init_convolution();
}