#include "common.h"
#include "NN.h"
#include "mymath.h"
#include "matrix.h"
// #include "adderData.h"

#define BITS 3

int main(){
    srand(time(NULL));
    // N bits adder

    // calculating the number of rows of the data matrix
    // input is 2 times the number of bits 
    // output is always 1 more than the number of bits
    int n = (1<<BITS);
    int rows = n*n;
    int inputCols = BITS*2;
    int outputCols = BITS+1;
    Mat data = matAlloc(rows,inputCols+outputCols);
    Mat input = {
        .rows = rows,
        .cols = inputCols,
        .stride = inputCols+outputCols,
        .data = data.data
    };
    Mat output = {
        .rows = rows,
        .cols = outputCols,
        .stride = inputCols+outputCols,
        .data = data.data+2*BITS
    };
    // Build the data matrix
    for(int x = 0 ; x < n ; ++x){
        for(int y = 0 ; y < n ; ++y){
            int z = x+y;
            for(int i = 0 ; i < BITS ; ++i){
                MAT(input,x*n+y,BITS-1-i) = (x>>i)&1;
                MAT(input,x*n+y,2*BITS-1-i) = (y>>i)&1;
            }
            for(int i = 0 ; i < BITS+1 ; ++i){
                MAT(output,x*n+y,BITS-i) = (z>>i)&1;
            }
        }
    }


    // 1 for the -1 to end the array, 2 for the input 
    // and output cols, BITS for the hidden layers
    int * arch = malloc(sizeof(int)*(1+2+BITS));
    arch[0] = inputCols;
    for(int i = 1 ; i <= BITS ; ++i){
        arch[i] = BITS*2;
    }
    arch[BITS+1] = outputCols;
    arch[BITS+2] = -1;

    NN nn = NNbuild(arch);
    NN  g = NNbuild(arch);
    // NN nn = NN_BUILD(inputCols,BITS*2,BITS*2,outputCols);
    // NN  g = NN_BUILD(inputCols,BITS*2,BITS*2,outputCols);
    
    nn.func=sigmoidf;
    NNrand(nn,-1,1);

    // SHOW_NN(nn,0);
    // 
    // printf("====================================\n");
    for(int i = 0 ; i < BITS*10000 ; ++i){
        // float c = cost(nn,input,output,sigmoidf);
        // printf("%d: cost = %f\n",i,c);
        backProp(nn,g,input,output);
        learn(nn,g,0.1f);
    }
    printf("====================================\n");
    SHOW_NN(nn,0);
    float c = cost(nn,input,output,sigmoidf);
    printf("final cost = %f\n",c);
    

}