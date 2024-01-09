#include "matrix.h"
#include "common.h"
#include "mymath.h"
typedef struct {
    int count;
    float w1_1;
    float w2_1;
    float b_1;

    float w1_2;
    float w2_2; 
    float b_2; 

    float w1_3; 
    float w2_3;
    float b_3;
}Xor;

float xor_data[] = {
    0, 0,   0,
    0, 1,   1,
    1, 0,   1,
    1, 1,   0
};

float foward(Xor m, float x1, float x2){
    float a = sigmoidf(m.w1_1*x1 + m.w2_1*x2 + m.b_1);
    float b = sigmoidf(m.w1_2*x1 + m.w2_2*x2 + m.b_2);
    return  sigmoidf(a*m.w1_3 + b*m.w2_3 + m.b_3);
}

void print_xor(Xor m, int indent){
    INDENT(indent);printf("Xor={\n");
    
    INDENT(indent+4);printf("w1_1:%4.4f\n",m.w1_1);
    INDENT(indent+4);printf("w2_1:%4.4f\n",m.w2_1);
    INDENT(indent+4);printf( "b_1:%4.4f\n",m.b_1);
    INDENT(indent+4);printf("\n");
    INDENT(indent+4);printf("w1_2:%4.4f\n",m.w1_2);
    INDENT(indent+4);printf("w2_2:%4.4f\n",m.w2_2);
    INDENT(indent+4);printf( "b_2:%4.4f\n",m.b_2);
    INDENT(indent+4);printf("\n");
    INDENT(indent+4);printf("w1_3:%4.4f\n",m.w1_3);
    INDENT(indent+4);printf("w2_3:%4.4f\n",m.w2_3);
    INDENT(indent+4);printf( "b_3:%4.4f\n",m.b_3);
    INDENT(indent);printf("}\n");
    // INDENT(indent);printf("x1w1_1 + x1w2_1 + b_1 = a\n");
    // INDENT(indent);printf("x2w1_2 + x2w2_2 + b_2 = b\n");
    // INDENT(indent);printf("aw1_3 + bw2_3 + b_3 = y\n");
}


int main(){
    FILE *fp = fopen("models/gates/xor.bin", "rb");
    Xor m;
    fread(&m, sizeof(Xor), 1, fp);

    for(int i = 0 ; i < 2 ; ++i){
        for(int j = 0 ; j < 2 ; ++j){
            printf("%d ^ %d = %f\n", i, j, foward(m, i, j));
        }
    }
    printf("====================================\n");
    for(int i = 0 ; i < 2 ; ++i){
        for(int j = 0 ; j < 2 ; ++j){
            printf("%d ^ %d = %f\n", i, j, roundf(foward(m, i, j)));
        }
    }
    print_xor(m, 0);
}