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

#define NN_INPUT(nn) ((Mat)(nn).a[0])
#define NN_OUTPUT(nn) ((Mat)(nn).a[(nn).count])


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
    for(int L = 1 ; L < layers ; ++L){
        nn.w[L-1] = matAlloc(nn.a[L-1].cols,architecture[L]);
        nn.b[L-1] = matAlloc(1,architecture[L]);
        nn.a[L] = matAlloc(1,architecture[L]);
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
    for(int L = 0; L < nn.count; L++){
        INDENT(indent);printf("layer %d = {\n",L);
        w=nn.w+L;
        b=nn.b+L;
        INDENT(indent+4);printf("w%d\e\e",L);
        SHOW_MAT(*w,indent+4);
        INDENT(indent+4);printf("b%d\e\e",L);
        SHOW_MAT(*b,indent+4);
        INDENT(indent);printf("}\n");
    }
}

void NNrand(NN nn, float low, float high){
    for(int L = 0; L < nn.count; L++){
        matRand(nn.w[L],low,high);
        matRand(nn.b[L],low,high);
    }
}

void foward(NN m, float (*func)(float)){
    for(int L = 0 ; L< m.count; L++){
        matDot(m.a[L+1],m.a[L],m.w[L]);
        matAdd(m.a[L+1],m.b[L]);
        matFunc(m.a[L+1],func);
    }
    // return MAT(m.a[m.count],0,0);
}

float cost(NN nn, Mat input, Mat expected, float (*func)(float)){

    int inprows = input.rows;
    int expcols = expected.cols;

    // assert(inprows == expected.rows);
    // assert(expcols == nn.a[nn.count].cols);
    

    float sum = 0.0f;
    for(int L = 0 ; L < inprows ; ++L){
        Mat x = matRow(input,L);
        Mat y = matRow(expected,L);
        
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
    // SHOW_NN(g,0);
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
    // SHOW_NN(g,0);
}

void saveNN(NN nn, char *filename){
    FILE *fp = fopen(filename,"wb");
    assert(fp != NULL);
    fwrite(&nn.count,sizeof(int),1,fp);
    fwrite(&nn.a[0].cols,sizeof(int),1,fp);
    for(int L = 0 ; L < nn.count ; ++L){
        fwrite(&nn.w[L].rows,sizeof(int),1,fp);
        fwrite(&nn.w[L].cols,sizeof(int),1,fp);
        fwrite(nn.w[L].data,sizeof(float),nn.w[L].rows*nn.w[L].cols,fp);
        
        fwrite(&nn.b[L].rows,sizeof(int),1,fp);
        fwrite(&nn.b[L].cols,sizeof(int),1,fp);
        fwrite(nn.b[L].data,sizeof(float),nn.b[L].rows*nn.b[L].cols,fp);
    }     
        

    fclose(fp);
}

NN loadNN(char *filename){
    FILE *fp = fopen(filename,"rb");
    assert(fp != NULL);
    NN nn;
    int count;
    fread(&count,sizeof(int),1,fp);
    nn.count = count;
    fread(&count,sizeof(int),1,fp);
    nn.w = (Mat*) malloc(sizeof(Mat)*nn.count);
    nn.b = (Mat*) malloc(sizeof(Mat)*nn.count);
    nn.a = (Mat*) malloc(sizeof(Mat)*(nn.count+1));
    nn.a[0] = matAlloc(1,count);

    for(int L = 0 ; L < nn.count ; ++L){
        int r,c;
        fread(&r,sizeof(int),1,fp);
        fread(&c,sizeof(int),1,fp);
        nn.w[L] = matAlloc(r,c);
        nn.w[L].data = malloc(sizeof(float)*r*c);
        fread(nn.w[L].data,sizeof(float),r*c,fp);

        fread(&r,sizeof(int),1,fp);
        fread(&c,sizeof(int),1,fp);
        nn.b[L] = matAlloc(r,c);
        nn.b[L].data = malloc(sizeof(float)*r*c);
        fread(nn.b[L].data,sizeof(float),r*c,fp);

        nn.a[L+1] = matAlloc(r,c);
        // printf("Layer:%d w:%dx%d b%dx%d\n",L,
        // nn.w[L].rows,
        // nn.w[L].cols,
        // nn.b[L].rows,
        // nn.b[L].cols );
    
    }
    return nn;
}


#endif // NN_H
