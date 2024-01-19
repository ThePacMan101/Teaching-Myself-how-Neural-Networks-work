#include "NN.h"
#include "matrix.h"
#include "mymath.h"
#include "common.h"

float xor_data[] = {
    0, 0,   0,
    0, 1,   1,
    1, 0,   1,
    1, 1,   0,
};

float (*func)(float) = sigmoidf;

int main(){
    // srand(time(NULL));
    srand(69);
    Mat input = {
        .rows = 4,
        .cols = 2,
        .stride = 3,
        .data = xor_data
    };

    Mat expected = {
        .rows = 4,
        .cols = 1,
        .stride = 3,
        .data = xor_data + 2
    };

    NN nn = NN_BUILD(2,2,1);
    nn.func=func;
    NN  g = NN_BUILD(2,2,1);
    NNrand(nn,0,1);
    float lr = 1e-1;

    printf("cost = %f\n",cost(nn,input,expected,func));
    SHOW_NN(nn,0);
    for(int i = 0 ; i < 1 ; ++i){
#if 0
        float eps = 1e-1;
        finiteDiff(nn,g,input,expected,eps,func);        
#else
        backProp(nn,g,input,expected);
#endif
        SHOW_NN(g,0);

    }

    for(int i = 0 ; i < 2 ; ++i){
        for(int j = 0 ; j < 2 ; ++j){
            MAT(NN_INPUT(nn),0,0) = i;
            MAT(NN_INPUT(nn),0,1) = j;
            foward(nn,func);
            printf("%d ^ %d = ",i,j);
            printf("%f\n",MAT(NN_OUTPUT(nn),0,0));
        }
    }
}