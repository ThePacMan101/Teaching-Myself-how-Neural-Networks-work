#include "common.h"
#include "matrix.h"
#include "mymath.h"
#include "NN.h"

float sample_data[] = {
    0, 0,   0, 1 ,      0, 0, 1,
    1, 0,   1, 0 ,      1, 0, 0,
    0, 1,   1, 0 ,      0, 1, 1,
    0, 1,   1, 1 ,      1, 0, 0,
    0, 0,   0, 0 ,      0, 0, 0,
    1, 1,   0, 0 ,      0, 1, 1,
    0, 1,   0, 0 ,      0, 0, 1,
    1, 1,   1, 0 ,      1, 0, 1,
    1, 0,   0, 1 ,      0, 1, 1,
    0, 1,   0, 1 ,      0, 1, 0,
    1, 1,   1, 1 ,      1, 1, 0
};  


float eps = 1e-1;
float lr = 1e-1;
float (*func)(float) = sigmoidf;

int main(){
    srand(time(NULL));
    Mat inputs = {
        .data = sample_data,
        .rows = 11,
        .cols = 4,
        .stride = 7
    };
    Mat expected = {
        .data = sample_data+4,
        .rows = 11,
        .cols = 3,
        .stride = 7
    };
    NN nn = NN_BUILD(4,2,2,2,3);
    NN g = NN_BUILD (4,2,2,2,3);
    NNrand(nn,-1,1);
    SHOW_NN(nn,0);
    printf("cost:%f\n",cost(nn,inputs,expected,func));
    printf("====================================\n");

    for(int epoch = 0 ; epoch < 1000000 ; ++epoch){
        g = finiteDiff(nn,g,inputs,expected,eps,func);
        learn(nn,g,lr);
        if(epoch%100000 == 0) printf("#");
    }

    printf("\ncost:%f\n",cost(nn,inputs,expected,func));
    printf("====================================\n");
    // for(int i = 0 ; i < 11 ; ++i){
    //     nn.a[0].data[0] = inputs.data[i];
    //     foward(nn,func);

    //     printf("%0.f%0.f + %0.f%0.f",MAT(inputs,i,0),MAT(inputs,i,1),MAT(inputs,i,2),MAT(inputs,i,3));
    //     printf(" = %0.f%0.f%0.f\n",MAT(expected,i,0),MAT(expected,i,1),MAT(expected,i,2));
    // }

    printf("====================================\n");

    Mat output = nn.a[nn.count];
    Mat row;

    for(int i = 0 ; i < 11 ; ++i){
        MAT(nn.a[0],0,0) = MAT(inputs,i,0);
        MAT(nn.a[0],0,1) = MAT(inputs,i,1);
        MAT(nn.a[0],0,2) = MAT(inputs,i,2);
        MAT(nn.a[0],0,3) = MAT(inputs,i,3);
        foward(nn,func);
        matFunc(output,roundf);
        row = matRow(inputs,i);
        SHOW_MAT(row,0);
        printf("\n == \n");
        SHOW_MAT(output,0);
        printf("\n------------------------------------\n\n");
        //printf("%f\t%f\t%f\n",MAT(output,i,0),MAT(output,i,1),MAT(output,i,2));
    }
    printf("final cost:%f\n",cost(nn,inputs,expected,func));
    printf("====================================\n");
    while(1){
        int a,b;
        printf("enter two numbers: ");
        scanf("%d %d",&a,&b);
        if(a == -1 || b==-1) break;
        MAT(nn.a[0],0,0) = a/2;
        MAT(nn.a[0],0,1) = a%2;
        MAT(nn.a[0],0,2) = b/2;
        MAT(nn.a[0],0,3) = b%2;
        foward(nn,func);
        printf("\n%d%d + %d%d\n == \n",
        a/2,
        a%2,
        b/2,
        b%2);

        SHOW_MAT(output,0);
    }

    return 0;
}