#include "common.h"
#include "matrix.h"
#include "mymath.h"
#include "NN.h"

// float sample_data[] = {
// /*   X   +   Y     =      Z     */
//     0,0,    0,0,        0,0,0,
//     0,0,    0,1,        0,0,1,
//     0,0,    1,0,        0,1,0,
//     0,0,    1,1,        0,1,1,
//     0,1,    0,0,        0,0,1,
//     0,1,    0,1,        0,1,0,
//     0,1,    1,0,        0,1,1,
//     0,1,    1,1,        1,0,0,
//     1,0,    0,0,        0,1,0,
//     1,0,    0,1,        0,1,1,
//     1,0,    1,0,        1,0,0,
//     1,0,    1,1,        1,0,1,
//     1,1,    0,0,        0,1,1,
//     1,1,    0,1,        1,0,0,
//     1,1,    1,0,        1,0,1,
//     1,1,    1,1,        1,1,0,
// };

// float another_data[] = {
//     00,    00,        000,
//     00,    01,        001,
//     00,    10,        010,
//     00,    11,        011,
//     01,    00,        001,
//     01,    01,        010,
//     01,    10,        011,
//     01,    11,        100,
//     10,    00,        010,
//     10,    01,        011,
//     10,    10,        100,
//     10,    11,        101,
//     11,    00,        011,
//     11,    01,        100,
//     11,    10,        101,
//     11,    11,        110,
// };

// float another_data[] = {
//     0,    0,        0,
//     0,    1,        1,
//     0,    2,        2,
//     0,    3,        3,
//     1,    0,        1,
//     1,    1,        2,
//     1,    2,        3,
//     1,    3,        4,
//     2,    0,        2,
//     2,    1,        3,
//     2,    2,        4,
//     2,    3,        5,
//     3,    0,        3,
//     3,    1,        4,
//     3,    2,        5,
//     3,    3,        6,
// };

float another_data[] = {
    0,0,    0,
    0,1,    1,
    1,0,    1,
    1,1,    0
};


// float another_data[]={
//     1, 3,
//     2, 6,
//     4, 12,
//     5, 15,
//     6, 18,
//     7, 21
// };

float eps = 1e-2;
float lr = 1e-1;
float (*func)(float) = sigmoidf; 

int main(){
    srand(time(NULL));
    Mat inputs = {
        .data = another_data,
        .rows = 4,
        .cols = 2,
        .stride = 3
    };
    Mat expected = {
        .data = another_data+2,
        .rows = 4,
        .cols = 1,
        .stride = 3
    };
    printf("\n====================================\n");
    NN nn = NN_BUILD(2,2,1);
    NN g = NN_BUILD (2,2,1);
    NNrand(nn,-5,5);
    SHOW_NN(nn,0);
    printf("cost:%f\n",cost(nn,inputs,expected,func));
    printf("====================================\n");

    for(int epoch = 0 ; epoch < 400000 ; ++epoch){
        g = finiteDiff(nn,g,inputs,expected,eps,func);
        //SHOW_NN(g,0);
        learn(nn,g,lr);
        if(epoch%40000 == 0) printf("#");
    }
    printf("\n");
    SHOW_NN(nn,0);
    float c = cost(nn,inputs,expected,func);
    printf("final cost:%f\n",c);
    printf("====================================\n");
    
    
    // if(c<1e-1){
    //     saveNN(nn,"models/times3.bin");
    //     printf("saved!\n");
    // }


    // while(1){
    //     int a,b;
    //     printf("enter two numbers: ");
    //     scanf("%d %d",&a,&b);
    //     if(a == -1 || b==-1) break;
    //     MAT(nn.a[0],0,0) = a/2;
    //     MAT(nn.a[0],0,1) = a%2;
    //     MAT(nn.a[0],0,2) = b/2;
    //     MAT(nn.a[0],0,3) = b%2;
    //     foward(nn,func);
    //     printf("\n%d%d + %d%d\n == \n",
    //     a/2,
    //     a%2,
    //     b/2,
    //     b%2);

    //     SHOW_MAT(output,0);
    // }
    // Mat output = nn.a[nn.count];
    // Mat row;

    // for(int i = 0 ; i < 16 ; ++i){
    //     MAT(nn.a[0],0,0) = MAT(inputs,i,0);
    //     MAT(nn.a[0],0,1) = MAT(inputs,i,1);
    //     MAT(nn.a[0],0,2) = MAT(inputs,i,2);
    //     MAT(nn.a[0],0,3) = MAT(inputs,i,3);
    //     foward(nn,func);
    //     matFunc(output,roundf);
    //     row = matRow(inputs,i);
    //     SHOW_MAT(row,0);
    //     printf("\n == \n");
    //     SHOW_MAT(output,0);
    //     printf("\n------------------------------------\n\n");
    //     //printf("%f\t%f\t%f\n",MAT(output,i,0),MAT(output,i,1),MAT(output,i,2));
    // }

    return 0;
}