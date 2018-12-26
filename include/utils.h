#ifndef __UTILS_H
#define __UTILS_H
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define LOG() printf("%s() at %s::%05d\n", __FUNCTION__, __FILE__, __LINE__);

double what_time_is_it_now();
void searchInDirectory(char *dirname);
float gaussianRandom(void);

#endif // __UTILS_H