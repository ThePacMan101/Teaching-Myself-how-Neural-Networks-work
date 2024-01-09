#include "matrix.h"
#include "common.h"
#include "mymath.h"


/*
  x1 ->  O\
           O -> y 
  x2 ->  O/
*/
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

float cost(Mat model, Xor m){
    float sum = 0.0f;
    for(int i = 0; i < model.rows; ++i){
        float x1 = MAT(model, i, 0);
        float x2 = MAT(model, i, 1);
        float y = foward(m,x1,x2);
        
        float d = y - MAT(model, i, 2);
        sum += d*d;
    }
    sum/=model.rows;
    return sum;
}

void rand_xor(Xor *m){
    m->w1_1 = rand_float()*10 - 5;
    m->w2_1 = rand_float()*10 - 5;
    m->b_1 = rand_float()*10 - 5;

    m->w1_2 = rand_float()*10 - 5;
    m->w2_2 = rand_float()*10 - 5;
    m->b_2 = rand_float()*10 - 5;

    m->w1_3 = rand_float()*10 - 5;
    m->w2_3 = rand_float()*10 - 5;
    m->b_3 = rand_float()*10 - 5;
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

Xor finiteDiff(Mat model, Xor m,float eps){
    
    Xor g;
    float c = cost(model,m);
    float saved;

    // ========================

    saved = m.w1_1;
    m.w1_1 += eps;
    g.w1_1 = (cost(model,m) - c)/eps;
    m.w1_1 = saved;

    saved = m.w2_1;
    m.w2_1 += eps;
    g.w2_1 = (cost(model,m) - c)/eps;
    m.w2_1 = saved;

    saved = m.b_1;
    m. b_1 += eps;
    g. b_1 = (cost(model,m) - c)/eps;
    m. b_1 = saved;

    // ========================

    saved = m.w1_2;
    m.w1_2 += eps;
    g.w1_2 = (cost(model,m) - c)/eps;
    m.w1_2 = saved;

    saved = m.w2_2;
    m.w2_2 += eps;
    g.w2_2 = (cost(model,m) - c)/eps;
    m.w2_2 = saved;

    saved = m.b_2;
    m. b_2 += eps;
    g. b_2 = (cost(model,m) - c)/eps;
    m. b_2 = saved;

    // ========================

    saved = m.w1_3;
    m.w1_3 += eps;
    g.w1_3 = (cost(model,m) - c)/eps;
    m.w1_3 = saved;

    saved = m.w2_3;
    m.w2_3 += eps;
    g.w2_3 = (cost(model,m) - c)/eps;
    m.w2_3 = saved;

    saved = m.b_3;
    m. b_3 += eps;
    g. b_3 = (cost(model,m) - c)/eps;
    m. b_3 = saved;
    
    return g;
}

Xor learn(Xor m, Xor g, float lr){
    m.w1_1 -= lr*g.w1_1;
    m.w2_1 -= lr*g.w2_1;
    m.b_1  -= lr*g.b_1;   
    m.w1_2 -= lr*g.w1_2;
    m.w2_2 -= lr*g.w2_2;
    m.b_2  -= lr*g.b_2;
    m.w1_3 -= lr*g.w1_3;
    m.w2_3 -= lr*g.w2_3;
    m.b_3  -= lr*g.b_3;
    
    return m;
}

float eps = 1e-4;
float lr = 1e-1;

int main(){
    srand(time(NULL));
    system("cls");
    Mat model = matAlloc(4,3);
    model.data = xor_data;

    Xor m,g;

    rand_xor(&m);
    print_xor(m,0);
    float c = cost(model,m);
    printf("====================================\n");
    printf("cost:%f\n",c);
    printf("====================================\n");
    

    int epoch =0 ;
    for(epoch = 0 ; epoch < 1000000 ;epoch++){
        g = finiteDiff(model,m,eps);
        m = learn(m,g,lr);
        // if(epoch % 10000 == 0){
        //     printf("epoch:%d cost: %f\n",epoch,cost(model,m));
        // }
    }
    
    c = cost(model,m);

    print_xor(m,0);
    printf("====================================\n");
    printf("cost:%f\n",c);
    printf("====================================\n");

    FILE *fp = fopen("models/gates/xor.bin","wb");
    fwrite(&m,sizeof(Xor),1,fp);
    fclose(fp);


    // g = finiteDiff(model,m,eps);
    // print_xor(g,0);

    

    // printf("====================================\n");
    // SHOW_MAT(model,4);


}