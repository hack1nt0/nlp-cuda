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
!    Construction of cubic spline with given second derivative coefficients
!    Example Program Text
!******************************************************************************/

#include <stdio.h>

#include "mkl.h"
#include "errcheck.inc"
#include "generatedata.inc"
#include "rescheck.inc"

#define N              7 // number of break points
#define NY             1 // number of functions
#define NIC          N-2 // number of internal conditions
#define NBC            2 // number of boundary conditions

#define NSCOEFF     (NY*(N-1)*DF_PP_CUBIC)  // total number of spline
                                            // coefficients

#define LEFT_LIMIT   1.0 // left limit of interpolation interval
#define RIGHT_LIMIT  3.0 // right limit of interpolation interval
#define FREQ         0.5


int main()
{
    DFTaskPtr task;                     // Data Fitting task descriptor
    MKL_INT nx;                         // number of break points
    MKL_INT xhint;                      // additional info about break points
    MKL_INT ny;                         // number of functions
    MKL_INT yhint;                      // additional info about function
    MKL_INT nscoeff;                    // number of spline coefficients
    MKL_INT scoeffhint;                 // additional info about spline
                                        // coefficients
    MKL_INT bc_type;                    // boundary conditions type
    MKL_INT nbc;                        // number of boundary conditions
    MKL_INT ic_type;                    // internal conditions type
    MKL_INT nic;                        // number of internal conditions

    float x[N];                         // array of break points
    float y[NY*N];                      // function values
    float ic[NIC];                      // array of internal conditions
    float bc[NBC];                      // array of boundary conditions
    float scoeff[NSCOEFF];              // array of spline coefficients

    MKL_INT stype, sorder;

    float left = LEFT_LIMIT, right = RIGHT_LIMIT;
    float freq;
    float left_val[N-1], right_val[N-1];
    float left_der2[N-1], right_der2[N-1];

    int i, j, errcode = 0;
    int errnums = 0;

    /***** Initializing parameters for Data Fitting task *****/

    sorder = DF_PP_CUBIC;
    stype  = DF_PP_DEFAULT;
    /***** Parameters describing interpolation interval *****/
    nx          = N;
    xhint       = DF_NON_UNIFORM_PARTITION;

    /***** Parameters describing function *****/
    ny          = NY;
    yhint       = 0;

    /***** Parameters describing spline coefficients storage *****/
    nscoeff     = NSCOEFF;
    scoeffhint  = DF_MATRIX_STORAGE_ROWS;

    /***** Parameters describing boundary conditions type *****/
    bc_type     = DF_BC_2ND_LEFT_DER | DF_BC_2ND_RIGHT_DER;
    nbc         = NBC;

    /***** Parameters describing internal conditions type *****/
    ic_type     = DF_IC_2ND_DER;
    nic         = NIC;

    /***** Generate array of uniformly distributed break points *****/
    errcode = sUniformRandSortedData( x, left, right, nx );
    CheckDfError(errcode);

    /***** Generate function y = sin(2 * Pi * freq * x) *****/
    errcode = sSinDataNotUniformGrid( y, x, FREQ, nx );
    CheckDfError(errcode);

    /***** Generate internal conditions needed for the spline
           construction *****/
    errcode = sSinDer2DataNotUniformGrid( ic, &x[1], FREQ, nic );
    CheckDfError(errcode);

    /***** Generate boundary conditions *****/
    errcode = sSinDer2DataNotUniformGrid( &bc[0], &x[0],    FREQ, 1 );
    CheckDfError(errcode);
    errcode = sSinDer2DataNotUniformGrid( &bc[1], &x[nx-1], FREQ, 1 );
    CheckDfError(errcode);

    /***** Create Data Fitting task *****/
    errcode = dfsNewTask1D( &task, nx, x, xhint, ny, y, yhint );
    CheckDfError(errcode);


    /***** Edit task parameters for cubic spline with provided 2nd derivatives
           construction *****/
    errcode = dfsEditPPSpline1D( task, sorder, stype, bc_type, bc, ic_type, ic,
                                 scoeff, scoeffhint );
    CheckDfError(errcode);


    /***** Construct cubic spline with provided 2nd derivatives
           using STD method *****/
    errcode = dfsConstruct1D( task, DF_PP_SPLINE, DF_METHOD_STD );
    CheckDfError(errcode);

    /***** Delete Data Fitting task *****/
    errcode = dfDeleteTask( &task );
    CheckDfError(errcode);

    /***** Check computed coefficients *****/

    /***** Check spline values in break points *****/
    errcode = sCheckCubBreakPoints( nx, x, ny, y, scoeff, left_val, right_val );
    if ( errcode < 0 ) errnums++;

    /***** Check that spline 2nd derivatives are equal for left
           and right piece of the spline for each break point *****/
    errcode = sCheckCub2ndDerConsistency( nx, x, ny, scoeff,
                                          left_der2, right_der2 );
    if ( errcode < 0 ) errnums++;

    /***** Check internal conditions *****/
    for( j = 0; j < nic; j++ )
    {
        if ( DF_ABS( ic[j] - left_der2[j] ) > EPSILON_SINGLE )
            errnums++;
    }

    /***** Check boundary conditions *****/
    errcode = sCheckCubBC( nx, x, ny, scoeff, bc_type, bc );
    if ( errcode < 0 ) errnums++;

    /***** Print results *****/
    printf("Number of break points : %d\n", (int)nx);

    /***** Print given function *****/
    printf("\n  X           Y(X)           Y\"(X)\n");

    printf(" %+lf   %+lf   %+lf\n", x[0], y[0], bc[0]);
    for( j = 1; j < nx-1; j++ )
    {
        printf(" %+lf   %+lf   %+lf\n", x[j], y[j], ic[j-1]);
    }
    printf(" %+lf   %+lf   %+lf\n", x[nx-1], y[nx-1], bc[1]);

    /***** Print computed spline coefficients *****/
    printf("\nCoefficients are calculated for a polynomial of the form:\n\n");
    printf("Pi(x) = Ai + Bi*(x - x(i)) + Ci*(x - x(i))^2 + Di*(x - x(i))^3\n");
    printf("    where x(i) <= x < x(i+1)\n");
    printf("\nSpline coefficients for Y:\n");
    printf(" i    Ai            Bi            Ci            Di        ");
    printf("    P(X[i])       P(X[i+1]) ");
    printf("    P\"(X[i])      P\"(X[i+1])\n");

    for( j = 0; j < nx-1; j++ )
    {
        printf(" %1d %+11.6f   %+11.6f   %+11.6f   %+11.6f   %+11.6f   %+11.6f",
                j, scoeff[4*j], scoeff[4*j + 1],
                scoeff[4*j + 2], scoeff[4*j + 3],
                right_val[j], left_val[j]);
        printf("   %+11.6f   %+11.6f\n",
                right_der2[j], left_der2[j]);
    }

    /***** Print summary of the test *****/
    if (errnums != 0)
    {
        printf("\n\nError: Computed default cubic spline coefficients");
        printf(" are incorrect\n");
        return 1;
    }
    else
    {
        printf("\n\nComputed default cubic spline coefficients");
        printf(" are correct\n");
    }

    return 0;
}
