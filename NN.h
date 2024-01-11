#ifndef NN_H
#define NN_H

#include "common.h"
#include "matrix.h"
#include "mymath.h"

// Neural Network structure
typedef struct{
    int count;  // Number of layers
    Mat *w;     // Array of matrices
    Mat *b;     // Array of biases
    Mat *a;     // Array of activations, there is 1 more than count
} NN;



#endif // NN_H
