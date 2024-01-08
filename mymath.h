#ifndef MYMATH_H
#define MYMATH_H

#include "common.h"

float sigmoidf(float x){
    return 1.0f / (1.0f + expf(-x));
}

float relu(float x){
    return x > 0 ? x : 0;
}

float drelu(float x){
    return x > 0 ? 1 : 0;
}

float dsigmoidf(float x){
    return x * (1.0f - x);
}

float tanhf(float x){
    return tanh(x);
}

float dtanhf(float x){
    return 1.0f - x * x;
}

float softmax(float x){
    return expf(x);
}   

float dsoftmax(float x){
    return 1.0f - x;
}   

#endif // MYMATH_H