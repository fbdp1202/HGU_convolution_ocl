#include <stdio.h>
#include <stdlib.h>
#include "image.h"

unsigned char* make_random_input(int input_size) {
	unsigned char* inputs = (unsigned char* )calloc(size, size(unsigned char))
	int i;
	for (i=0; i<inputs; i++)
		inputs[i] = (unsigned char)(rand()%256);
	return inputs;
}

void delete_input(unsigned char* inputs) {
	free(inputs);
}