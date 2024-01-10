#ifndef MATRIX_H
#define MATRIX_H

#include "common.h"


typedef struct{
    int rows;
    int cols;
    int stride;
    float *data;
} Mat;
// a matrix is a struct with the number of rows, columns, the stride 
// (the number of columns in the original matrix) and a pointer to the data


#define MAT(m, r, c) *(m.data + (r) * (m).stride + (c))
// MAT(m, r, c) is a pointer to the element at row r and column c of matrix m

#define SHOW_MAT(m,indent) \
    do { \
        INDENT(indent); \
        printf("%s = {\n",(#m)); \
        matShow(m, indent+4); \
        INDENT(indent); \
        printf("}\n"); \
    } while(0)
// show the matrix m with indentation indent showing the name of the matrix

#define Iprintf_f(n,arg) \
    for(int k = 0; k < indent; k++) \
        printf(" "); \
    printf("%f     ", arg);
// _f stands for float here. Usefull if I ever need a matrix of ints, doubles or something else

void matCopy(Mat src, Mat dst){
    assert(src.rows == dst.rows && src.cols == dst.cols);
    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            MAT(dst, i, j) = MAT(src, i, j);
        }
    }
}


// Starts a matrix
Mat matAlloc(int rows, int cols){
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.data = (float*)calloc(rows * cols, sizeof(float));
    assert(m.data != NULL);
    return m;
}

// Fills a matrix with a value
void matFill(Mat m, float val){
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            MAT(m, i, j) = val;
        }
    }
}

// Applies a function to every element of a matrix
void matFunc(Mat m, float (*func)(float)){
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            MAT(m, i, j) = (*func)(MAT(m, i, j));
        }
    }
}

// Randomizes every element of a matrix between low and high
void matRand(Mat m, float low, float high){
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            MAT(m, i, j) = RAND(low, high);
        }
    }
}

// Prints a matrix with identation
void matShow(Mat m, int indent){
    char str[10];
    for(int i = 0; i < m.rows; i++){
        INDENT(indent);printf("|");
        for(int j = 0; j < m.cols; j++){
            sprintf(str,"%8g|", MAT(m, i, j));
            printf("%s",str);
        }
        printf("\n");
    }
}

//Can be used standalone, but most of the time it'll be used by the macros above





//======================================================
//                  Matrix operations
//======================================================


// dot product
void matDot(Mat result , Mat matrixA , Mat matrixB){ 
    //
    //  using: ROWxCOL
    //            NxM             PxQ          NxQ
    //        |... , ...|     |... , ...|      |... , ...|
    //        |... , ...|  *  |... , ...|  ==  |... , ...|
    //        |   ...   |     |   ...   |      |   ...   |
    //            
    //        IF M != P, the operation is not possible
    //
    //      e.g.:
    //           3x1               3x2
    //           |a|   1x2         |ax , ay|
    //           |b| * |x , y| ==  |bx , by|
    //           |c|               |cx , cy|
    //
    assert(matrixA.cols == matrixB.rows && matrixA.rows == result.rows && matrixB.cols == result.cols);    
    for(int i = 0; i < matrixA.rows; i++){
        for(int j = 0; j < matrixB.cols; j++){
            MAT(result, i, j) = 0;
            for(int k = 0; k < matrixA.cols; k++){
                MAT(result, i, j) += MAT(matrixA, i, k) * MAT(matrixB, k, j);
            }
        }
    }
}


// addition
void matAdd(Mat dest , Mat src){
    assert(src.rows == dest.rows && src.cols == dest.cols);

    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            MAT(dest, i, j) += MAT(src, i, j);
        }
    }
}



#endif // MATRIX_H
