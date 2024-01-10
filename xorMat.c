#include "matrix.h"
#include "common.h"
#include "mymath.h"

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

// Neural Network structure
typedef struct{
    int count;  // Number of layers
    Mat *w;     // Array of matrices w[0] represents the matrix of the first layer
    Mat *b;     // Array of biases:  b[0] represents the bias of the first layer   
} NN;

float xor_data[] = {
    0, 0,   0,
    0, 1,   1,
    1, 0,   1,
    1, 1,   0
};

int main(){
    Mat model = matAlloc(1000,3);
    //model.data = xor_data;

}