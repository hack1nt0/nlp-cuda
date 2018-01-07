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
!    Calculation of Akima cubic spline coefficients and integral
!    computation with callback function  Example Program Text
!******************************************************************************/

#include <stdio.h>

#include "mkl.h"
#include "errcheck.inc"
#include "generatedata.inc"
#include "rescheck.inc"

#define N             10 /* number of breakpoints */
#define NY             1 /* number of datasets to interpolate */
#define NLIM           1 /* number of pairs of integration limits */

#define NSCOEFF        (N-1)*DF_PP_CUBIC

#define LLIM_X         0.0  // left  limit of interpolation interval
#define RLIM_X         3.0  // right limit of interpolation interval
#define LLIM_INTEGR   -0.5  // left  limit of integration interval
#define RLIM_INTEGR    3.5  // right limit of integration interval
#define FREQ    0.75

/*******************************************************************************
!   Definition of the integration call back for integral calculations
! on the interval (-inf, x[0]).
!
! API
!   int  left_akima_integr( MKL_INT64* n, MKL_INT64 lcell[], double llim[],
!                           MKL_INT64 rcell[], double rlim[], double r[],
!                           void *x0 )
!
! INPUT PARAMETERS:
!  n      - number of pairs of integration limits
!  lcell  - array of size n with indices of the cells that contain
!           the left-side integration limits
!  llim   - array of size n that holds the left-side integration limits
!  rcell  - array of size n with indices of the cells that contain
!           the right-side integration limits
!  rlim   - array of size n that holds the right-side integration limits
!  x0     - left  limit of interpolation interval
!
! OUTPUT PARAMETERS:
!  r      - array of integration results
!
! RETURN VALUE:
!  The status returned by the callback function:
!   0  - indicates successful completion of the callback operation
!   <0 - error
!   >0 - warning
*******************************************************************************/
int  left_akima_integr( MKL_INT64* n, MKL_INT64 lcell[], double llim[],
                        MKL_INT64 rcell[], double rlim[], double r[], void *x0 );

/*******************************************************************************
!   Definition of the integration call back for integral calculations
! on the interval [x[N-1], +inf).
!
! API
!   int right_akima_integr( MKL_INT64* n, MKL_INT64 lcell[], double llim[],
!                           MKL_INT64 rcell[], double rlim[], double r[],
!                           void *xN )
!
! INPUT PARAMETERS:
!  n      - number of pairs of integration limits
!  lcell  - array of size n with indices of the cells that contain
!           the left-side integration limits
!  llim   - array of size n that holds the left-side integration limits
!  rcell  - array of size n with indices of the cells that contain
!           the right-side integration limits
!  rlim   - array of size n that holds the right-side integration limits
!  xN     - right limit of interpolation interval
!
! OUTPUT PARAMETERS:
!  r      - array of integration results
!
! RETURN VALUE:
!  The status returned by the callback function:
!   0  - indicates successful completion of the callback operation
!   <0 - error
!   >0 - warning
*******************************************************************************/
int right_akima_integr( MKL_INT64* n, MKL_INT64 lcell[], double llim[],
                        MKL_INT64 rcell[], double rlim[], double r[], void *xN );

int main()
{
    DFTaskPtr task;                       // Data Fitting task descriptor
    MKL_INT sorder;                       // spline order
    MKL_INT stype;                        // spline type
    MKL_INT nx;                           // number of break points
    MKL_INT xhint;                        // additional info about break points
    MKL_INT ny;                           // number of functions
    MKL_INT yhint;                        // functions storage format
    MKL_INT scoeffhint;                   // additional info about spline
                                          // coefficients
    MKL_INT bc_type;                      // boundary conditions type
    MKL_INT ic_type;                      // internal conditions type
    MKL_INT nlim;                         // number of pairs of integration
                                          // limits
    MKL_INT llimhint, rlimhint;           // integration limits storage formats
    MKL_INT rhint;                        // integration results storage format
    double left = LLIM_X, right = RLIM_X; // limits of interpolation interval
    double x[N];                          // break points
    double y[N*NY];                       // function values
    double bc[] = { 0.0, -0.5 };          // boundary conditions
    double *ic;                           // internal conditions
    double scoeff[NSCOEFF];               // Akima spline coefficients
    double llim = LLIM_INTEGR;            // left  limit of integration interval
    double rlim = RLIM_INTEGR;            // right limit of integration interval
    double *ldatahint, *rdatahint;        // additional info about the
                                          // integration limits
    double r;                             // integration results

    dfdIntegrCallBack le_cb, re_cb;       // integration call backs
    double le_params, re_params;          // integration call backs parameters

    double freq;
    double r_ref;
    double left_val[NY*(N-1)], right_val[NY*(N-1)];
    double left_der1[NY*(N-1)], right_der1[NY*(N-1)];

    int i, j, errcode = 0;
    int errnums = 0;

    /***** Initializing parameters for Data Fitting task *****/

    /***** Parameters describing order and type of the spline *****/
    sorder       = DF_PP_CUBIC;
    stype        = DF_PP_AKIMA;

    /***** Parameters describing interpolation interval *****/
    nx           = N;
    xhint        = DF_NON_UNIFORM_PARTITION;

    /***** Parameters describing function *****/
    ny           = NY;
    yhint        = DF_NO_HINT;

    /***** Parameter describing additional info about spline
           coefficients *****/
    scoeffhint   = DF_NO_HINT;

    /***** Parameters describing boundary conditions type *****/
    bc_type      = DF_BC_2ND_LEFT_DER | DF_BC_1ST_RIGHT_DER;

    /***** Parameters describing internal conditions type *****/
    /* No internal conditions are provided for Akima cubic spline */
    ic_type      = DF_NO_IC;
    ic = 0;

    /***** Parameters decsribing integration limits *****/
    nlim         = NLIM;
    llimhint     = DF_NO_HINT;
    rlimhint     = DF_NO_HINT;

    /***** Additional information about the structure of integration
           limits *****/
    /* No additional info is provided */
    ldatahint = 0;
    rdatahint = 0;

    /***** Parameter dascribing integration results storage format *****/
    rhint = DF_NO_HINT;

    /***** Generate partition with uniformly distributed break points *****/
    errcode = dUniformRandSortedData( x, left, right, nx );
    CheckDfError(errcode);

    /***** Call backs for integration on the outer intervals *****/
    le_cb =  left_akima_integr;
    re_cb = right_akima_integr;
    /* Use limits of interpolation interval as call back parameters */
    le_params = left;
    re_params = right;

    /***** Generate function y = sin(2 * Pi * freq * x) *****/
    freq = FREQ;

    errcode = dSinDataNotUniformGrid( y, x, freq, nx );
    CheckDfError(errcode);

    /***** Create Data Fitting task *****/
    errcode = dfdNewTask1D( &task, nx, x, xhint, ny, y, yhint );
    CheckDfError(errcode);

    /***** Edit task parameters for Akima cubic spline construction *****/
    errcode = dfdEditPPSpline1D( task, sorder, stype, bc_type, bc,
                                 ic_type, ic, scoeff, scoeffhint );
    CheckDfError(errcode);

    /***** Construct Akima cubic spline using STD method *****/
    errcode =  dfdConstruct1D( task, DF_PP_SPLINE, DF_METHOD_STD );
    CheckDfError(errcode);

    /***** Compute integral for the spline on the interval (llim, rlim) *****/
    errcode = dfdIntegrateEx1D( task, DF_METHOD_PP, nlim, &llim, llimhint,
                                &rlim, rlimhint, ldatahint, rdatahint,
                                &r, rhint, le_cb, &le_params, re_cb, &re_params,
                                0, 0, 0, 0 );
    CheckDfError(errcode);

    /***** Check computed coefficients *****/

    /***** Check spline values in break points *****/
    errcode = dCheckCubBreakPoints( nx, x, ny, y, scoeff, left_val, right_val );
    if ( errcode < 0 ) errnums++;

    /***** Check that spline 1st derivatives are equal for left
           and right piece of the spline for each break point *****/
    errcode = dCheckCub1stDerConsistency( nx, x, ny, scoeff,
                                          left_der1, right_der1 );
    if ( errcode < 0 ) errnums++;


    /***** Print results *****/
    printf("Number of break points : %d\n", (int)nx);

    /***** Print given function *****/
    printf("\n i  x(i)        y(i)\n");

    for( j = 0; j < nx; j++ )
    {
        printf(" %1d %+lf   %+lf\n", j, x[j], y[j]);
    }


    /***** Print computed spline coefficients *****/
    printf("\nCoefficients are calculated for a polynomial of the form:\n\n");
    printf("Pi(x) = Ai + Bi*(x - x(i)) + Ci*(x - x(i))^2 + Di*(x - x(i))^3\n");
    printf("    where x(i) <= x < x(i+1)\n");
    printf("\nSpline coefficients for Y:\n");
    printf(" i      Ai              Bi              Ci              Di      ");
    printf("      Pi(x(i))      Pi(x(i+1))    Pi'(x(i))     Pi'(x(i+1))\n");

    for( j = 0; j < nx-1; j++ )
    {
        printf(" %1d %+13.6f   %+13.6f   %+13.6f   %+13.6f   %+11.6f   %+11.6f",
                j, scoeff[sorder*j],     scoeff[sorder*j + 1],
                scoeff[sorder*j + 2], scoeff[sorder*j + 3],
                right_val[j], left_val[j]);
        printf("   %+11.6f   %+11.6f\n", right_der1[j], left_der1[j]);
    }

    /***** Print computed integration results *****/
    printf("\nSpline-based integral on interval [ %4.1lf, %4.1lf ) is %lf\n",
            llim, rlim, r );

    /***** Delete Data Fitting task *****/
    errcode = dfDeleteTask( &task );
    CheckDfError(errcode);

    /***** Print summary of the test *****/
    if (errnums != 0) {
        printf("\n\nError: Computed Akima cubic spline coefficients");
        printf(" or integartion results are incorrect\n");
        return 1;
    }
    else {
        printf("\n\nComputed Akima cubic spline coefficients");
        printf(" and integration results are correct\n");
    }

    return 0;
}

/*******************************************************************************
!   Integration call back for integral calculations on the interval
!   (-inf, x[0]).
!
! API
!   int  left_akima_integr( MKL_INT64* n, MKL_INT64 lcell[], double llim[],
!                           MKL_INT64 rcell[], double rlim[], double r[],
!                           void *x0 )
!
! INPUT PARAMETERS:
!  n      - number of pairs of integration limits
!  lcell  - array of size n with indices of the cells that contain
!           the left-side integration limits
!  llim   - array of size n that holds the left-side integration limits
!  rcell  - array of size n with indices of the cells that contain
!           the right-side integration limits
!  rlim   - array of size n that holds the right-side integration limits
!  x0     - left  limit of interpolation interval
!
! OUTPUT PARAMETERS:
!  r      - array of integration results
!
! RETURN VALUE:
!  The status returned by the callback function:
!   0  - indicates successful completion of the callback operation
!   <0 - error
!   >0 - warning
*******************************************************************************/
int left_akima_integr( MKL_INT64* n, MKL_INT64 lcell[], double llim[],
                       MKL_INT64 rcell[], double rlim[], double r[], void *x0 )
{
    MKL_INT64 i;
    double *x = (double*)x0;

    for ( i = 0; i < n[0]; i++ )
    {
        r[i] = x[0] * ( x[0] - llim[i] );
    }

    return 0;
}

/*******************************************************************************
!   Integration call back for integral calculations on the interval
!   [x[N-1], +inf).
!
! API
!   int right_akima_integr( MKL_INT64* n, MKL_INT64 lcell[], double llim[],
!                           MKL_INT64 rcell[], double rlim[], double r[],
!                           void *xN )
!
! INPUT PARAMETERS:
!  n      - number of pairs of integration limits
!  lcell  - array of size n with indices of the cells that contain
!           the left-side integration limits
!  llim   - array of size n that holds the left-side integration limits
!  rcell  - array of size n with indices of the cells that contain
!           the right-side integration limits
!  rlim   - array of size n that holds the right-side integration limits
!  xN     - right limit of interpolation interval
!
! OUTPUT PARAMETERS:
!  r      - array of integration results
!
! RETURN VALUE:
!  The status returned by the callback function:
!   0  - indicates successful completion of the callback operation
!   <0 - error
!   >0 - warning
*******************************************************************************/
int right_akima_integr( MKL_INT64* n, MKL_INT64 lcell[], double llim[],
                        MKL_INT64 rcell[], double rlim[], double r[], void *xN )
{
    MKL_INT64 i;
    double *x = (double*)xN;
    for ( i = 0; i < n[0]; i++ )
    {
        r[i] = x[0] * ( rlim[i] - x[0] );
    }

    return 0;
}
