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
!    mkl_zomatcopy2 - out-of-place transposition routine,
!    Example Program Text ( C Interface )
!******************************************************************************/
#include <mkl_trans.h>
#include "common_func.h"

int main(int argc, char *arcv[])
{ 
  size_t n=4, m=6; /* rows, cols of source matrix */
  MKL_Complex16 alpha;
  MKL_Complex16 src[]={ 
         1.,  2., 0., 0.,   3.,  4.,  0., 0.,    5., 6.,  0., 0.,
        13., 14., 0., 0.,   15., 16., 0., 0.,   17., 18., 0., 0.,
        25., 26., 0., 0.,   27., 28., 0., 0.,   29., 30., 0., 0.,
        37., 38., 0., 0.,   39., 40., 0., 0.,   41., 42., 0., 0.
      }; /* source matrix */
  MKL_Complex16 dst[12]; /* destination matrix */
  alpha.real = 1.;
  alpha.imag = 0.;

  printf("\nExample of using mkl_zomatcopy2 transposition\n");
  printf("INPUT DATA:\nSource matrix A:\n");
  print_matrix(n, m, 'z', src);

  printf("Destination matrix - copy of meaningful part with transposition:\n");
  /* 
  **  Copy of meaningful part of source matrix with transposition
  */
  mkl_zomatcopy2('R'    /* row-major ordering */, 
                 'T'    /* A will be transposed */, 
                  4     /* rows */, 
                  3     /* cols */, 
                  alpha /* scales the input matrix */, 
                  src   /* source matrix */, 
                  6     /* src_row */, 
                  2     /* scr_col */, 
                  dst   /* destination matrix */, 
                  4     /* dst_row */, 
                  1     /* dst_col */);
  print_matrix(3, 4, 'z', dst);
  /* New matrix: dst = { 
  **    1.,  2., 13., 14., 25., 26., 37., 38., 
  **    3.,  4., 15., 16., 27., 28., 39., 40.,
  **    5.,  6., 17., 18., 29., 30.  41., 42.
  ** }
  */

  printf("Destination matrix - copy of submatrix(3,4) with conjugate transposition:\n");
  /*
  **  Copy of meaningful part of source matrix with transposition
  */
  mkl_zomatcopy2('R'    /* row-major ordering */, 
                 'C'    /* matrix will be transposed */, 
                  4     /* rows */, 
                  3     /* cols */, 
                  alpha /* scales the input matrix */, 
                  src   /* source matrix */, 
                  6     /* src_row */, 
                  1     /* scr_col */, 
                  dst   /* destination matrix */, 
                  4     /* dst_row */, 
                  1     /* dst_col */);
  print_matrix(3, 4, 'z', dst);
  /*  New matrix: dst = { 
  **     1.,  -2.,  13., -14.,  25., -26.,  37., -38.,
  **     0.,   0.,   0.,   0.,   0.,   0.,   0.,  0.,
  **     3.,  -4.,  15., -16.,  27., -28.,  39., -40.
  **  }
  */

  return 0;
}
