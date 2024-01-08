#include "NN.h"

int main(){
    float or_data[] = {
        0, 0,   0,
        0, 1,   1,
        1, 0,   1,
        1, 1,   1
    };

    Mat input  = {4, 2, 3,  or_data   }; 
    Mat output = {4, 1, 3,  or_data+2 };
    Mat fulldata   = {4, 3, 3,  or_data };

    printf("matrices ={\n");
        SHOW_MAT(input,4);
        SHOW_MAT(output,4);
        SHOW_MAT(fulldata,4);
    printf("}\n");
}