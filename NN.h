#ifndef NN_H
#define NN_H

#include "common.h"
#include "matrix.h"
#include "mymath.h"

// Neural Network structure
typedef struct{
    int count;  // Number of layers
    Mat *w;     // Array of matrices
    Mat *b;     // Array of biases
    Mat *a;     // Array of activations, there is 1 more than count
} NN;


// exemple use: NNbuild((int[]){ 2 , 2 , 1 ,-1});
// builds: 2 input neurons -> 3 hidden neurons -> 1 output neuron
// all neurons are fully connected
NN NNbuild(int *architecture){
    /*  For achitecture == {2 , 2 , 1}
        
        we have nn.count = 2, because we don't count the input layer

        we have inputs: x1 x2, so a0 = |x1 , x2| which is a 1x2 matrix
        
        1st Layer:
                We have 2 neurons, so w1 is a 2x2 matrix 
                and b1 is a 1x2 matrix

                therefore a1 is a 1x2 matrix
        2nd Layer:
                We have 1 neuron, so w2 is a 2x1 matrix
                and b2 is a 1x1 matrix

                therefore a2 is a 1x1 matrix
    */
    NN nn;
    int layers = 0;
    while(architecture[layers] != -1) layers++;
    assert(layers > 0);
    nn.count = layers-1;

    nn.w = (Mat*)calloc(nn.count,sizeof(Mat));
    assert(nn.w != NULL);        
    nn.b = (Mat*)calloc(nn.count,sizeof(Mat));      
    assert(nn.b != NULL);
    nn.a = (Mat*)calloc(nn.count+1,sizeof(Mat));
    assert(nn.a != NULL);

    nn.a[0] = matAlloc(1,architecture[0]); 
    // a0 is a matrix 1xN where N is the first number in the architecture
    // a in: {2 , 2 , 1} we have a0 = |x1 , x2| which is a 1x2 matrix, representing our inputs

    //Now we have to loop through the architecture array to build the rest of the NN
    for(int i = 1 ; i < layers ; ++i){
        nn.w[i-1] = matAlloc(nn.a[i-1].cols,architecture[i]);
        nn.b[i-1] = matAlloc(1,architecture[i]);
        nn.a[i] = matAlloc(1,architecture[i]);
    }

    return nn;
}

#define NN_BUILD(...) NNbuild((int[]){__VA_ARGS__, -1})

#define SHOW_NN(nn,indent) \
    do { \
        INDENT(indent); \
        printf("%s = {\n",(#nn)); \
        NNprint(nn, indent+4); \
        INDENT(indent); \
        printf("}\n"); \
    } while(0)

void NNprint(NN nn,int indent){
    Mat *w,*b;
    for(int i = 0; i < nn.count; i++){
        INDENT(indent);printf("layer %d = {\n",i);
        w=nn.w+i;
        b=nn.b+i;
        INDENT(indent+4);printf("w%d\e\e",i);
        SHOW_MAT(*w,indent+4);
        INDENT(indent+4);printf("b%d\e\e",i);
        SHOW_MAT(*b,indent+4);
        INDENT(indent);printf("}\n");
    }
}

void NNrand(NN nn, float low, float high){
    for(int i = 0; i < nn.count; i++){
        matRand(nn.w[i],low,high);
        matRand(nn.b[i],low,high);
    }
}

float foward(NN m, float (*func)(float)){
    for(int i = 0 ; i< m.count; i++){
        matDot(m.a[i+1],m.a[i],m.w[i]);
        matAdd(m.a[i+1],m.b[i]);
        matFunc(m.a[i+1],func);
    }
    return MAT(m.a[m.count],0,0);
}

float cost(NN nn, Mat input, Mat expected, float (*func)(float)){

    int inprows = input.rows;
    int expcols = expected.cols;

    // assert(inprows == expected.rows);
    // assert(expcols == nn.a[nn.count].cols);
    

    float sum = 0.0f;
    for(int i = 0 ; i < inprows ; ++i){
        Mat x = matRow(input,i);
        Mat y = matRow(expected,i);
        
        matCopy(nn.a[0],x);
        foward(nn,func);

        for(int j = 0 ; j < expcols ; ++j){
            sum += pow(MAT(nn.a[nn.count],0,j) - MAT(y,0,j),2);
        }

    }
    sum/=inprows;
    return sum;
}

NN finiteDiff(NN nn , NN g, Mat input, Mat expected, float eps,float (*func)(float)){
    float saved;
    float c = cost(nn,input,expected,func);

    //ITERATE THROUGH EACH LAYER OF THE NETWORK
    for(int L = 0 ; L < nn.count ; ++L){
        
        //ITERATE THROUGH EACH WEIGHT OF THE LAYER
        for(int i = 0 ; i < nn.w[L].rows ; ++i){
            for(int j = 0 ; j < nn.w[L].cols ; ++j){
                saved = MAT(nn.w[L],i,j);   //save the current value
                MAT(nn.w[L],i,j) += eps;    //wiggle the value
                MAT(g.w[L],i,j) = (cost(nn,input,expected,func) - c)/eps;    //add the finite diff to the gradient
                MAT(nn.w[L],i,j) = saved;   //restore the value
            }   
        }
        //ITERATE THROUGH EACH BIAS OF THE LAYER
        for(int i = 0 ; i < nn.b[L].rows ; ++i){
            for(int j = 0 ; j < nn.b[L].cols ; ++j){
                saved = MAT(nn.b[L],i,j);   //save the current value
                MAT(nn.b[L],i,j) += eps;    //wiggle the value
                MAT(g.b[L],i,j) = (cost(nn,input,expected,func) - c)/eps;   //add the finite diff to the gradient
                MAT(nn.b[L],i,j) = saved;  //restore the value
            }   
        }
       
    }

    return g;
}

void learn(NN nn, NN g, float lr){
    for(int L = 0 ; L < nn.count ; ++L){
        
        //ITERATE THROUGH EACH WEIGHT OF THE LAYER
        for(int i = 0 ; i < nn.w[L].rows ; ++i){
            for(int j = 0 ; j < nn.w[L].cols ; ++j){
                MAT(nn.w[L],i,j) -= lr * MAT(g.w[L],i,j);   //gradient descent
            }   
        }
        //ITERATE THROUGH EACH BIAS OF THE LAYER
        for(int i = 0 ; i < nn.b[L].rows ; ++i){
            for(int j = 0 ; j < nn.b[L].cols ; ++j){
               MAT(nn.b[L],i,j) -= lr * MAT(g.b[L],i,j);   //gradient descent
            }   
        }
       
    }
}

#endif // NN_H
