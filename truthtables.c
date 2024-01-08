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


float cost(Mat output, Mat target){
    assert(output.rows == target.rows);
    assert(output.cols == target.cols);
    float sum = 0;
    for(int i = 0; i < output.rows; i++){
        for(int j = 0; j < output.cols; j++){
            sum += pow(MAT(output, i, j) - MAT(target, i, j), 2);
        }
    }
    return sum;
}

// void forward(Mat input, Mat output, float w1, float w2, float b){
//     assert(input.cols == 2);
//     assert(output.cols == 1);
//     assert(input.rows == output.rows);
//     for(int i = 0; i < input.rows; i++){
//         float x1 = MAT(input, i, 0);
//         float x2 = MAT(input, i, 1);
//         float y  = MAT(output, i, 0);
//         y = x1*w1 + x2*w2 + b;
//         MAT(output, i, 0) = y;
//     }
// }

// void learn(Mat input, Mat output, Mat expected, float w1, float w2, float b, float lr){
//     assert(input.cols == 2);
//     assert(output.cols == 1);
//     assert(input.rows == output.rows);
//     assert(input.rows == expected.rows);
//     assert(output.rows == expected.rows);
//     for(int i = 0; i < input.rows; i++){
//         float x1 = MAT(input, i, 0);
//         float x2 = MAT(input, i, 1);
//         float y  = MAT(output, i, 0);
//         float t  = MAT(expected, i, 0);
//         float d  = y - t;
//         w1 -= lr * d * x1;
//         w2 -= lr * d * x2;
//         b  -= lr * d;
//     }
// }

int main(){
    srand(10);
    system("cls");
    Mat model = matAlloc(4,3);
    model.data = and_data;
    
    Mat input    = {4, 2, 3,  model.data   }; 
    Mat output   = {4, 1, 3,  model.data+2 };

    Mat grad = matAlloc(4,1);
    
    Mat expected = matAlloc(4,1);
    matCopy(output, expected);

    matRand(output, -1, 1);

    float w1 = rand()/(float)(RAND_MAX)*2 - 1;
    float w2 = rand()/(float)(RAND_MAX)*2 - 1;
    float b  = rand()/(float)(RAND_MAX)*2 - 1;



    SHOW_MAT(input,0);
    SHOW_MAT(expected,4);
    SHOW_MAT(output,4);

    printf("=====================================\n");
    printf("w1:%f\n", w1);
    printf("w2:%f\n", w2); 
    printf("b:%f\n", b);
    printf("=====================================\n");

    printf("cost before:%f\n", cost(output, expected));

    // learn(input, output, expected, w1, w2, b, 0.1);
    
    printf("cost after:%f\n", cost(output, expected));

    SHOW_MAT(model,4);

    // printf("matrices ={\n");
    //     SHOW_MAT(input,4);
    //     SHOW_MAT(output,4);
    //     SHOW_MAT(fulldata,4);
    // printf("}\n");
}