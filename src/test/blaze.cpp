//
// Created by DY on 17-10-13.
//

#include <vector>
#include <blaze/math/CustomMatrix.h>
#include <iostream>
#include <blaze/Math.h>

using blaze::CustomMatrix;
using blaze::unaligned;
using blaze::unpadded;

typedef CustomMatrix<int,unaligned,unpadded>  CustomType;

int main() {

    using blaze::StaticVector;
    using blaze::DynamicVector;

// Instantiation of a static 3D column vector. The vector is directly initialized as
//    ( 4 -2  5 )
    StaticVector<int,3UL> a{ 4, -2, 5 };

// Instantiation of a dynamic 3D column vector. Via the subscript operator the values are set to
//    ( 2  5 -3 )
    DynamicVector<int> b( 3UL );
    b[0] = 2;
    b[1] = 5;
    b[2] = -3;

// Adding the vectors a and b
    DynamicVector<int> c = a + b;

// Printing the result of the vector addition
    std::cout << "c =\n" << c << "\n";
    return 0;
}
