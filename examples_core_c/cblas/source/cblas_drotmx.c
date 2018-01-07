/*******************************************************************************
* Copyright 1999-2015 Intel Corporation All Rights Reserved.
*
* The source code,  information  and material  ("Material") contained  herein is
* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
* Material  contains  proprietary  information  of  Intel or  its suppliers  and
* licensors.  The Material is protected by  worldwide copyright  laws and treaty
* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
* in any way without Intel's prior express written permission.  No license under
* any patent,  copyright or other  intellectual property rights  in the Material
* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
* property rights must be express and approved by Intel in writing.
*
* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
* suppliers or licensors in any way.
*******************************************************************************/

/*
!  Content:
!      C B L A S _ D R O T M   Example Program Text ( C Interface )
!******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"

#include "mkl_example.h"

int main(int argc, char *argv[])
{
      FILE *in_file;
      char *in_file_name;

      MKL_INT    n, incx, incy;
      double     param[5];
      double    *x, *y;
      MKL_INT    len_x, len_y;

      printf("\n     C B L A S _ D R O T M  EXAMPLE PROGRAM\n");

/*       Get input parameters                                  */

      if( argc == 1 ) {
         printf("\n You must specify in_file data file as 1-st parameter");
         return 1;
      }
      in_file_name = argv[1];

/*       Get input data                                       */

      if( (in_file = fopen( in_file_name, "r" )) == NULL ) {
         printf("\n ERROR on OPEN '%s' with mode=\"r\"\n", in_file_name);
         return 1;
      }
      if( GetIntegerParameters(in_file, &n) != 1 ) {
          printf("\n ERROR of n reading\n");
          return 1;
      }
      if( GetIntegerParameters(in_file, &incx, &incy) != 2 ) {
          printf("\n ERROR of incx, incy reading\n");
          return 1;
      }
      if( GetVectorD(in_file, 5, param, 1) != 5 ) {
        printf("\n ERROR of vector PARAM reading\n");
        return 1;
      }

      len_x = 1+(n-1)*abs(incx);
      len_y = 1+(n-1)*abs(incy);
      x    = (double *)mkl_calloc(len_x, sizeof( double ), 64);
      y    = (double *)mkl_calloc(len_y, sizeof( double ), 64);
      if( x == NULL || y == NULL ) {
          printf( "\n Can't allocate memory for arrays\n");
          return 1;
      }
      if( GetVectorD(in_file, n, x, incx) != len_x ) {
        printf("\n ERROR of vector X reading\n");
        return 1;
      }
      if( GetVectorD(in_file, n, y, incy) != len_y ) {
        printf("\n ERROR of vector Y reading\n");
        return 1;
      }
      fclose(in_file);

/*       Print input data                                      */

      printf("\n     INPUT DATA");
      printf("\n       N=%d", n);
      PrintVectorD(1, 5, param, 1, "PARAM");
      PrintVectorD(FULLPRINT, n, x, incx, "X");
      PrintVectorD(FULLPRINT, n, y, incy, "Y");

/*      Call CBLAS_DROTM subroutine ( C Interface )            */

      cblas_drotm(n, x, incx, y, incy, param);

/*       Print output data                                     */

      printf("\n\n     OUTPUT DATA");
      PrintVectorD(FULLPRINT, n, x, incx, "X");
      PrintVectorD(FULLPRINT, n, y, incy, "Y");

      mkl_free(x);
      mkl_free(y);

      return 0;
}
