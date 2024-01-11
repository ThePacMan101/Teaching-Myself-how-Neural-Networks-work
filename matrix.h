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

void matCopy(Mat dst, Mat src){
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
    //┌─┐│
    char str[50];
    INDENT(indent);
    for(int c = 0 ; c < m.cols ; ++c){
        printf("+");
        for(int i=0 ; i < 8 ; ++i) printf("-");
    }
    printf("+\n");
    for(int i = 0; i < m.rows; i++){
        INDENT(indent);printf("|");
        for(int j = 0; j < m.cols; j++){
            sprintf(str,"%8g", MAT(m, i, j));
            for(int c=0;c<8;c++)
                printf("%c",str[c]);
            printf("|");
        }
        printf("\n");
    }
    INDENT(indent);
    for(int c = 0 ; c < m.cols ; ++c){
        printf("+");
        for(int i=0 ; i < 8 ; ++i) printf("-");
    }
    printf("+\n");
}

//Can be used standalone, but most of the time it'll be used by the macros above

Mat matRow(Mat m, int row){
    
    assert((row < m.rows) && (row >= 0));
    Mat r;
    r.data =&MAT(m, row, 0);
    r.cols = m.cols;
    r.stride = m.stride;
    r.rows = 1;

    return r;
    
}

Mat matCol(Mat m, int col){
    assert((col < m.cols) && (col >= 0));
    Mat c;
    c.data = &MAT(m, 0, col);
    c.cols = 1;
    c.stride = m.stride;
    c.rows = m.rows;
    
    return c;
}




//======================================================
//                  Matrix operations
//======================================================


// dot product
void matDot(Mat result , Mat a , Mat b){ 
    //
    //  using: ROWxCOL
    //            NxM             PxQ          NxQ
    //        +---------+     +---------+      +---------+
    //        |... , ...|     |... , ...|      |... , ...|
    //        |... , ...|  *  |... , ...|  ==  |... , ...|
    //        |   ...   |     |   ...   |      |   ...   |
    //        +---------+     +---------+      +---------+
    //            
    //        IF M != P, the operation is not possible
    //
    //      e.g.:
    //           3x1               3x2
    //           +-+   1x2         +---+---+   
    //           |a|   +--+--+     |ax , ay|
    //           |b| * |x , y| ==  |bx , by|
    //           |c|   +--+--+     |cx , cy|
    //           +-+               +---+---+   
    assert(a.cols == b.rows && a.rows == result.rows && b.cols == result.cols);    
    for(int i = 0; i < a.rows; i++){
        for(int j = 0; j < b.cols; j++){
            MAT(result, i, j) = 0;
            for(int k = 0; k < a.cols; k++){
                MAT(result, i, j) += MAT(a, i, k) * MAT(b, k, j);
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
