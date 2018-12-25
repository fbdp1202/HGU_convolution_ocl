#ifndef __UTILS_H
#define __UTILS_H
#include <stdio.h>
#include <math.h>

#define LOG_FUN() printf("%s() at %s::%05d\n", __FUNCTION__, __FILE__, __LINE__);

float gaussianRandom(void);

#endif // __UTILS_H