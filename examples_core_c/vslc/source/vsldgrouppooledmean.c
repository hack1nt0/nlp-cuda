/*******************************************************************************
* Copyright 2003-2015 Intel Corporation All Rights Reserved.
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
!    Calculation of group/pooled means Example Program Text
!******************************************************************************/

#include <stdio.h>

#include "mkl.h"
#include "errcheck.inc"
#include "generatedata.inc"
#include "statchars.inc"

#define DIM     3      /* Task dimension */
#define N       10000  /* Number of observations */
#define G       2      /* Number of groups */
#define GN      2      /* Number of group means */

#define P_THRESHOLD    0.005

double C[DIM][DIM] = {
    { 1.0, 0.0, 0.0 },
    { 0.0, 1.0, 0.0 },
    { 0.0, 0.0, 1.0 }
};

double m[DIM] = { 0.0, 0.0, 0.0 };

/***** 1st and 2nd group means to be returned *****/
MKL_INT group_mean_indices[G] = { 1, 1 };

int main()
{
    VSLSSTaskPtr task;
    MKL_INT dim=DIM, n=N, x_storage;
    double x[DIM*N], pld_mean[DIM], grp_mean[DIM*GN];
    MKL_INT group_indices[N];

    double a = 0.0, sigma = 1.0;

    int i, j, errcode, ret_value;
    int errnums = 0;

    double pval_pld_mean[DIM], pval_grp_mean[DIM*GN];

    /***** Generate data set using VSL Gaussian RNG with mean a = 0 and
           stdev = 1 *****/
    errcode = dGenerateGaussianData( x, dim, n, a, sigma );
    CheckVslError(errcode);

    /***** Initializing parameters for Summary Statistics task *****/
    dim              = DIM;
    n                = N;
    x_storage        = VSL_SS_MATRIX_STORAGE_ROWS;

    /***** Create Summary Statistics task *****/
    errcode = vsldSSNewTask( &task, &dim, &n, &x_storage, x, 0, 0 );
    CheckVslError(errcode);

    /***** Dividing elements into odd and even *****/
    for(i = 0; i < n; i++)
    {
        group_indices[i] = i % 2;
    }

    /***** Initialization of the task parameters for pooled and
           group mean estimators *****/
    errcode = vsldSSEditPooledCovariance( task, group_indices,
        pld_mean, 0, group_mean_indices, grp_mean, 0 );
    CheckVslError(errcode);

    /***** Compute group and pooled mean using 1PASS method  *****/
    errcode = vsldSSCompute( task,VSL_SS_POOLED_MEAN | VSL_SS_GROUP_MEAN,
                                  VSL_SS_METHOD_1PASS );
    CheckVslError(errcode);

    /***** Testing stat characteristics of means *****/
    /* Compute p-values for group mean estimates */
    dComputePvalsMean( dim, n, grp_mean, m, (double*)C, pval_grp_mean );
    dComputePvalsMean( dim, n, &grp_mean[dim], m, (double*)C,&pval_grp_mean[dim] );
    /* Compute p-values for pooled mean estimates */
    dComputePvalsMean( dim, n, pld_mean, m, (double*)C, pval_pld_mean );

    for(i = 0; i < dim; i++)
    {
        if (pval_grp_mean[i] < P_THRESHOLD) errnums++;
        if (pval_grp_mean[i + dim] < P_THRESHOLD) errnums++;
        if (pval_pld_mean[i] < P_THRESHOLD) errnums++;
    }

    /***** Printing results *****/
    printf("Task dimension : %d\n", dim);
    printf("Number of observations : %d\n", n);

    /***** Print exact mean *****/
    printf("\nExact means:\n");
    for(i = 0; i < dim; i++)
    {
        printf("%+lf ", m[i]);
    }

    /***** Print group mean estimates *****/
    printf("\nGroup means:\n");
    for(i = 0; i < dim; i++)
    {
        printf("%+lf ", grp_mean[i]);
    }

    printf("     ");

    for(i = 0; i < dim; i++)
    {
        printf("%+lf ", grp_mean[i + dim]);
    }

    /***** Print pooled mean estimates *****/
    printf("\nPooled means:\n");
    for(i = 0; i < dim; i++)
    {
        printf("%+lf ", pld_mean[i]);
    }

    /***** Print P-values of the group means *****/
    printf("\nP-values of the computed group mean:\n");
    for(i = 0; i < dim; i++)
    {
        printf("%+lf ", pval_grp_mean[i]);
    }

    printf("     ");

    for(i = 0; i < dim; i++)
    {
        printf("%+lf ", pval_grp_mean[i + dim]);
    }

    /***** Printing P-values of the pooled mean *****/
    printf("\nP-values of the computed pooled mean:\n");
    for(i = 0; i < dim; i++)
    {
        printf("%+lf ", pval_pld_mean[i]);
    }

    /***** Printing summary of the test *****/
    if (errnums == 0) {
        printf("\n\nPooled and group mean estimates ");
        printf("are agreed with theory\n");
        ret_value = 0;
    }
    else {
        printf("\n\nPooled and group mean estimates ");
        printf("are disagreed with theory\n");
        ret_value = 1;
    }

    /***** Delete Summary Statistics task *****/
    errcode = vslSSDeleteTask( &task );
    CheckVslError(errcode);

    MKL_Free_Buffers();

    return ret_value;
}
