#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>

#define INDENT(n)   for(int k = 0; k < n; k++) printf(" ")
// general use macro for indentation
#define RAND(low,high)   ((float)rand()/(float)(RAND_MAX)) * ((high) - (low)) + (low)
// randomize float number between low and high

#endif // COMMON_H