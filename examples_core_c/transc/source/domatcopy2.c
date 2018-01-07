/*******************************************************************************
* Copyright 2010-2015 Intel Corporation All Rights Reserved.
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
!    mkl_domatcopy2 - out-of-place transposition routine,
!    Example Program Text ( C Interface )
!******************************************************************************/
#include <mkl_trans.h>
#include "common_func.h"

int main(int argc, char *argv[])
{ 
  size_t n=4, m=6; /* rows, cols of source matrix */
  double src[]= { 
    1,  0,  2,  0,  3,  0,
    4,  0,  5,  0,  6,  0,
    7,  0,  8,  0,  9,  0,
    10, 0,  11, 0,  12, 0
  }; /* source matrix */
  double dst[20]; /* destination matrix */

  printf("\nExample of using mkl_domatcopy2 transposition\n");

  printf("INPUT DATA:\nSource matrix A:\n");
  print_matrix(n, m, 'd', src);

  printf("Destination matrix - copy of me meaningful part with transposition:\n");
  /*
  **  Copy of meaningful part of source matrix with transposition
  */
  mkl_domatcopy2('R'  /* row-major ordering */, 
                 'T'  /* A will be transposed */, 
                  4   /* rows */, 
                  5   /* cols */, 
                  1   /* scales the input matrix */, 
                  src /* source matrix */, 
                  6   /* src_row */, 
                  2   /* scr_col */, 
                  dst /* destination matrix */, 
                  4   /* dst_row */, 
                  1   /* dst_col */);
  print_matrix(3, 4, 'd', dst);
  /* New matrix: dst = { 
  **      1,  4,  7,  10,
  **      2,  5,  8,  11,
  **      3,  6,  9,  12
  **    }
  */

  return 0;
}
