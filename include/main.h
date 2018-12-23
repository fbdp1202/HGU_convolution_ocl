#ifndef MAIN_API
#define MAIN_API
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct layer{
	int k;
	int c;
	int w;
	int h;
	float *weights;
	float *outputs;
}layer;


#endif // MAIN_API