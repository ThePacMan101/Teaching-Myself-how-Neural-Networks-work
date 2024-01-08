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

Mat matAlloc(int rows, int cols){
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.data = (float*)calloc(rows * cols, sizeof(float));
    return m;
}

void matFill(Mat m, float val){
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            MAT(m, i, j) = val;
        }
    }
}

void matFunc(Mat m, float (*func)(float)){
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            MAT(m, i, j) = (*func)(MAT(m, i, j));
        }
    }
}

void matShow(Mat m, int indent){
    for(int i = 0; i < m.rows; i++){
        INDENT(indent);printf("|");
        for(int j = 0; j < m.cols; j++){
            printf("%*.*f\t|",4,3, MAT(m, i, j));
            //Iprintf_f(4,MAT(m, i, j));
        }
        printf("\n");
    }
}

void matRand(Mat m, float low, float high){
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            MAT(m, i, j) = RAND(low, high);
        }
    }
}
//Can be used standalone, but most of the time it'll be used by the macros above





//======================================================
//                  Matrix operations
//======================================================


// dot product
void matDot(Mat a, Mat b, Mat c){
    assert(a.cols == b.rows && a.rows == c.rows && b.cols == c.cols);

    for(int i = 0; i < a.rows; i++){
        for(int j = 0; j < b.cols; j++){
            MAT(c, i, j) = 0;
            for(int k = 0; k < a.cols; k++){
                MAT(c, i, j) += MAT(a, i, k) * MAT(b, k, j);
            }
        }
    }
}


// addition
void matAdd(Mat a, Mat b, Mat c){
    assert(a.rows == b.rows && a.cols == b.cols && a.rows == c.rows && a.cols == c.cols);

    for(int i = 0; i < a.rows; i++){
        for(int j = 0; j < a.cols; j++){
            MAT(c, i, j) = MAT(a, i, j) + MAT(b, i, j);
        }
    }
}



#endif // MATRIX_H
