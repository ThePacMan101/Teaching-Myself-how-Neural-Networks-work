#include "common.h"
#include "matrix.h"
#include "mymath.h"
#include "NN.h"

float sample_data[] = {
    0,0,    0,
    0,1,    1,
    1,0,    1,
    1,1,    0
};

int main(){
    NN nn;
    Mat input = {
        .rows = 4,
        .cols = 2,
        .stride = 3,
        .data = sample_data
    };
    Mat expected = {
        .rows = 4,
        .cols = 1,
        .stride = 3,
        .data = sample_data+2
    };
    nn = loadNN("models/gates/xor.bin");
    SHOW_NN(nn,0);
    printf("====================================\n");
    
    

    for(int i = 0 ; i < 2 ; i++)
        for(int j = 0 ; j < 2 ; j++){
            MAT(nn.a[0],0,0) = i;
            MAT(nn.a[0],0,1) = j;
            foward(nn,sigmoidf);
            printf("i ^ j = %f\n",MAT(nn.a[nn.count],0,0));
        }

    for(int L = 0 ; L < nn.count+1 ; ++L){
        SHOW_MAT(nn.a[L],0);
    }
    
}