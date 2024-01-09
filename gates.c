#include "matrix.h"
#include "common.h"
#include "mymath.h"

float or_data[] = {
    0, 0,   0,
    0, 1,   1,
    1, 0,   1,
    1, 1,   1
};

float nor_data[] = {
    0, 0,   1,
    0, 1,   0,
    1, 0,   0,
    1, 1,   0
};
        
float and_data[] = {
    0, 0,   0,
    0, 1,   0,
    1, 0,   0,
    1, 1,   1
};
                
float nand_data[] = {
    0, 0,   1,
    0, 1,   1,
    1, 0,   1,
    1, 1,   0
};

float xor_data[] = {
    0, 0,   0,
    0, 1,   1,
    1, 0,   1,
    1, 1,   0
};

float imp_data[] = {
    0, 0,   1,
    0, 1,   1,
    1, 0,   0,
    1, 1,   1
};

float cost(Mat model, float w1, float w2, float b){
    float sum = 0.0f;
    for(int i = 0; i < model.rows; ++i){
        float x1 = MAT(model, i, 0);
        float x2 = MAT(model, i, 1);
        float y = sigmoidf(x1*w1 + x2*w2 + b);
        
        float d = y - MAT(model, i, 2);
        sum += d*d;
    }
    sum/=model.rows;
    return sum;
}

int main(){
    srand(time(NULL));
    system("cls");
    Mat model = matAlloc(4,3);
    model.data = imp_data;


    float w1 = rand_float()*10 - 5;
    float w2 = rand_float()*10 - 5;
    float b = rand_float()*10 - 5;

    printf("=====================================\n");
    printf("w1:%f\n", w1);
    printf("w2:%f\n", w2);
    printf("b:%f\n", b); 
    printf("=====================================\n");



    float eps = 1e-1;
    float lr = 1e-1;
    int epoch =0 ;
    for(epoch = 0 ; epoch < 1000000 ;epoch++){
        float c =  cost(model,w1,w2,b);
        float dw1 = (cost(model,w1+eps,w2,b) - c)/eps;
        float dw2 = (cost(model,w1,w2+eps,b) - c)/eps;
        float db = (cost(model,w1,w2,b+eps) - c)/eps;

        w1 -= lr * dw1;
        w2 -= lr * dw2;
        b  -= lr * db;

        if(c<1e-4)break;
    }
    printf("cost:%f w1:%f w2:%f b:%f \n",cost(model,w1,w2,b), w1, w2,b);
    printf("epochs:%d\n",epoch);
    printf("=====================================\n");

    printf("0 0 %f\n",sigmoidf(MAT(model,0,0)*w1+MAT(model,0,1)*w2+b));
    printf("0 1 %f\n",sigmoidf(MAT(model,1,0)*w1+MAT(model,1,1)*w2+b));
    printf("1 0 %f\n",sigmoidf(MAT(model,2,0)*w1+MAT(model,2,1)*w2+b));
    printf("1 1 %f\n",sigmoidf(MAT(model,3,0)*w1+MAT(model,3,1)*w2+b));
    

    SHOW_MAT(model,4);
}