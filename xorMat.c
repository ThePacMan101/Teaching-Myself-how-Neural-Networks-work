#include "matrix.h"
#include "common.h"
#include "mymath.h"
#include "NN.h"

/*
==========================================================

Passing trough a layer looks like this:

    actFunc(    |a1 , a2| * | w1 | + | b |   )  =  Y
                            | w2 |

    actFunc(X*W1 + B1) = Y -> 
    actFunc(Y*W2 + B2) = Z -> 
    actFunc(Z*W3 + B3) = [...]

==========================================================

    in our case (from Xor.c):

    when X*W1 + B1 = Y, we had:
      I.    x1*w1_1 + x2*w2_1 + b_1 = a1
     II.    x1*w1_2 + x2*w2_2 + b_2 = a2
    III.    a1*w1_3 + a2*w2_3 + b_3 = Y

    So, we can rewrite the equations above as:
    
    
   --> First layer:
    1x2         2x2               1x2                                                  1x2
    |x1 , x2| * |w1_1 , w1_2|  =  |x1*w1_1 + x2*w2_1 , x2*w1_2 + x2*w2_2|   =   invsig(|a1 - b_1 , a2 - b_2|)   
                |w2_1 , w2_2|          

    1x2                                       1x2                      1x2
    |x1*w1_1 + x2*w2_1 , x2*w1_2 + x2*w2_2| + |b_1 , b_2|   =   invsig(|a1 , a2|)

               1x2               1x2 
    sig(invsig(|a1 , a2|))   =   |a1 , a2|

  -->  Second layer:
    1x2         2x1          1x1                               1x1
    |a1 , a2| * |w1_3|   =   |a1*w1_3 + a2*w2_3|   =   invsig(|Y - b_3|) 
                |w2_3|
            
    1X1                   1X1                1X1       
    |a1*w1_3 + a2*w2_3| + |b_3|   =   invsig(|Y|)

    sig(invsig|Y|) = |Y| == OUTPUT
               

==========================================================

    
*/

float xor_data[] = {
    0, 0,   0,
    0, 1,   1,
    1, 0,   1,
    1, 1,   0
};

float id[] = {
    1,0,
    0,1
};

typedef struct {
    int count;    // Number of layers: for now is fixed at 2
    Mat as[3];    // Array of activations, there is 1 more than count
    Mat ws[2];    // Array of matrices
    Mat bs[2];    // Array of biases
} Xor;

void randXor(Xor *xor){
    for(int i = 0; i < xor->count; i++){
        matRand(xor->ws[i],0,1);
        matRand(xor->bs[i],0,1);
    }

}

void printXor(Xor *xor, int indent){
    Mat *w,*b;
    INDENT(indent);printf("Xor = {\n");	
    for(int i = 0; i < xor->count; ++i){
        INDENT(indent+4);
        printf("w%d\e\e",i);
        w=xor->ws+i;
        SHOW_MAT(*w,indent+4);
        b=xor->bs+i;
        INDENT(indent+4);
        printf("b%d\e\e",i);
        SHOW_MAT(*b,indent+4);
    }
    INDENT(indent);printf("}\n");
}

float foward(Xor m){
    for(int i = 0 ; i< m.count; i++){
        matDot(m.as[i+1],m.as[i],m.ws[i]);
        matAdd(m.as[i+1],m.bs[i]);
        matFunc(m.as[i+1],sigmoidf);
    }
    return MAT(m.as[m.count],0,0);
}

float cost(Xor m, Mat input, Mat expected){

    int inprows = input.rows;
    int expcols = expected.cols;

    assert(inprows == expected.rows);
    assert(expcols == m.as[m.count].cols);
    

    float sum = 0.0f;
    for(int i = 0 ; i < inprows ; ++i){
        Mat x = matRow(input,i);
        Mat y = matRow(expected,i);
        
        matCopy(m.as[0],x);
        foward(m);

        for(int j = 0 ; j < expcols ; ++j){
            sum += pow(MAT(m.as[m.count],0,j) - MAT(y,0,j),2);
        }

    }
    sum/=inprows;
    return sum;
}

Xor finiteDiff(Xor m , Xor g, Mat input, Mat expected, float eps){
    float saved;
    float c = cost(m,input,expected);

    //ITERATE THROUGH EACH LAYER OF THE NETWORK
    for(int L = 0 ; L < m.count ; ++L){
        
        //ITERATE THROUGH EACH WEIGHT OF THE LAYER
        for(int i = 0 ; i < m.ws[L].rows ; ++i){
            for(int j = 0 ; j < m.ws[L].cols ; ++j){
                saved = MAT(m.ws[L],i,j);   //save the current value
                MAT(m.ws[L],i,j) += eps;    //wiggle the value
                MAT(g.ws[L],i,j) = (cost(m,input,expected) - c)/eps;    //add the finite diff to the gradient
                MAT(m.ws[L],i,j) = saved;   //restore the value
            }   
        }
        //ITERATE THROUGH EACH BIAS OF THE LAYER
        for(int i = 0 ; i < m.bs[L].rows ; ++i){
            for(int j = 0 ; j < m.bs[L].cols ; ++j){
                saved = MAT(m.bs[L],i,j);   //save the current value
                MAT(m.bs[L],i,j) += eps;    //wiggle the value
                MAT(g.bs[L],i,j) = (cost(m,input,expected) - c)/eps;   //add the finite diff to the gradient
                MAT(m.bs[L],i,j) = saved;  //restore the value
            }   
        }
       
    }

    return g;
}

Xor xorAlloc(){
    Xor m;
    m.count = 2;
    m.as[0] = matAlloc(1,2);
    m.ws[0] = matAlloc(2,2);
    m.bs[0] = matAlloc(1,2);
    m.as[1] = matAlloc(1,2);
    m.ws[1] = matAlloc(2,1);
    m.bs[1] = matAlloc(1,1);
    m.as[2] = matAlloc(1,1);
    return m;
}

void learn(Xor m, Xor g, float lr){
    for(int L = 0 ; L < m.count ; ++L){
        
        //ITERATE THROUGH EACH WEIGHT OF THE LAYER
        for(int i = 0 ; i < m.ws[L].rows ; ++i){
            for(int j = 0 ; j < m.ws[L].cols ; ++j){
                MAT(m.ws[L],i,j) -= lr * MAT(g.ws[L],i,j);   //gradient descent
            }   
        }
        //ITERATE THROUGH EACH BIAS OF THE LAYER
        for(int i = 0 ; i < m.bs[L].rows ; ++i){
            for(int j = 0 ; j < m.bs[L].cols ; ++j){
               MAT(m.bs[L],i,j) -= lr * MAT(g.bs[L],i,j);   //gradient descent
            }   
        }
       
    }
}

float eps = 1e-3;
float lr = 1e-1;

int main(){
    system("cls");
    srand(time(NULL));
    Mat inputs = {
        .data = xor_data,
        .rows = 4,
        .cols = 2,
        .stride = 3
    };
    Mat outputs = {
        .data = xor_data+2,
        .rows = 4,
        .cols = 1,
        .stride = 3
    };

    Xor m = xorAlloc();
    Xor g = xorAlloc();
    
    randXor(&m);

    


    printXor(&m,0);
    printf("cost:%f\n",cost(m,inputs,outputs));
    printf("====================================\n");
    printf("LEARNING...\n");

    for(int epoch = 0 ; epoch < 100000 ; ++epoch){
        finiteDiff(m,g,inputs,outputs,eps);
        learn(m,g,lr);
        if(epoch%5000 == 0) printf("#");
    }
    printf("\n====================================\n");
    printXor(&m,0);
    printf("cost:%f\n",cost(m,inputs,outputs));
    printf("====================================\n");
    for(int i = 0 ; i < 2 ; i ++)
        for(int j = 0 ; j < 2 ; j ++){
            MAT(m.as[0],0,0) = i;
            MAT(m.as[0],0,1) = j;
            printf("%i ^ %i = %f\n",i,j,foward(m));
        }

}