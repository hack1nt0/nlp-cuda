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
!    Computation of robust covariance matrix and mean Example Program Text
!******************************************************************************/

#include <stdio.h>

#include "mkl.h"
#include "errcheck.inc"
#include "generatedata.inc"
#include "statchars.inc"

#define DIM         5       /* Task dimension */
#define N           5000    /* Number of observations */

#define P_THRESHOLD 0.001

#define RATIO       2       /* Ratio of outliers in the dataset */
#define M           100.0   /* Mean of the outliers */
#define COEFF       1.0     /* Coefficient to compute covarince of outliers */

/***** Robust method parameters *****/
#define BD_POINT    0.4
#define ARP         0.001
#define ACCURACY    0.001
#define ITER_NUM    5

/***** Parameters for major distribution *****/
float C[DIM][DIM] = {
    { 1.0, 0.1, 0.1, 0.1, 0.1 },
    { 0.1, 2.0, 0.1, 0.1, 0.1 },
    { 0.1, 0.1, 1.0, 0.1, 0.1 },
    { 0.1, 0.1, 0.1, 2.0, 0.1 },
    { 0.1, 0.1, 0.1, 0.1, 1.0 }
};

float a[DIM] = { 0.0, 0.0, 0.0, 0.0, 0.0 };

int main()
{
    VSLSSTaskPtr task;
    MKL_INT dim;
    MKL_INT n;
    MKL_INT x_storage;
    MKL_INT cov_storage;
    MKL_INT rcov_storage;
    float x[DIM][N];
    float mean[DIM], rmean[DIM];
    float cov[DIM*(DIM+1)/2], rcov[DIM*(DIM+1)/2];
    float pval_c[DIM][DIM], pval_r[DIM][DIM];
    int i, j, k, errcode;
    int errnums = 0;

    MKL_INT robust_params_n;
    float robust_method_params[VSL_SS_TBS_PARAMS_N];

    /***** Initializing parameters for Summary Statistics task *****/
    dim              = DIM;
    n                = N;
    x_storage        = VSL_SS_MATRIX_STORAGE_ROWS;
    cov_storage      = VSL_SS_MATRIX_STORAGE_L_PACKED;
    rcov_storage     = VSL_SS_MATRIX_STORAGE_L_PACKED;

    /***** Generate data set *****/
    errcode = sGenerateContaminatedDataset( (float*)x, dim, n, a,
                                            (float*)C, RATIO, M, COEFF );
    CheckVslError(errcode);

    /***** Create Summary Statistics task *****/
    errcode = vslsSSNewTask( &task, &dim, &n, &x_storage, (float*)x, 0, 0 );
    CheckVslError(errcode);

    errcode = vslsSSEditCovCor( task, mean, cov, &cov_storage, 0, 0 );

    /***** Initialization of the task parameters
           for robust covariance estimator *****/
    robust_params_n         = VSL_SS_TBS_PARAMS_N;
    robust_method_params[0] = BD_POINT;
    robust_method_params[1] = ARP;
    robust_method_params[2] = ACCURACY;
    robust_method_params[3] = ITER_NUM;

    errcode =  vslsSSEditRobustCovariance( task, &rcov_storage,
                                           &robust_params_n,
                                           robust_method_params,
                                           rmean, rcov );
    CheckVslError(errcode);

    /***** Compute covariance matrix using FAST method  *****/
    errcode = vslsSSCompute( task, VSL_SS_COV |
                             VSL_SS_ROBUST_COV,
                             VSL_SS_METHOD_FAST | VSL_SS_METHOD_TBS );
    CheckVslError(errcode);

    /***** Printing results *****/
    printf("Task dimension : %d\n", dim);
    printf("Number of observations : %d\n\n", n);

    /***** Print original mean and covariance matrix *****/
    printf("Original covariance matrix\n");
    for(i = 0; i < dim; i++)
    {
        for(j = 0; j < dim; j++)
        {
            printf("%lf ", C[i][j]);
        }
        printf("\n");
    }

    printf("\nOriginal vector of means\n");
    for(i = 0; i < dim; i++)
    {
        printf("%lf, ", a[i]);
    }
    printf("\n\n");

    /***** Print classical mean and covariance matrix estimate *****/
    printf("Classical covariance estimate\n");
    k = 0;
    for ( i = 0; i < dim; i++ )
    {
        for(j = 0; j <= i; j++)
        {
            printf("%lf ", cov[k++]);
        }
        printf("\n");
    }

    printf("\nClassical mean estimate\n");
    for (i = 0; i < dim; i++)
    {
        printf("%lf, ", mean[i]);
    }
    printf("\n\n");

    /***** Print robust mean and covariance matrix estimate *****/
    printf("Robust covariance estimate:\n");
    k = 0;
    for (i = 0; i < dim; i++)
    {
        for(j = 0; j <= i; j++)
        {
            printf("%lf ", rcov[k++]);
        }
        printf("\n");
    }

    printf("\nRobust mean estimate:\n");
    for(i = 0; i < dim; i++)
    {
        printf("%lf, ", rmean[i]);
    }
    printf("\n");

    /***** Testing stat characteristics of classic and robust
           covariance matrices *****/
    sComputePvalsVariance( dim, n,  cov, cov_storage, (float*)C,
                           (float*)pval_c );
    sComputePvalsVariance( dim, n, rcov, rcov_storage, (float*)C,
                           (float*)pval_r );
    sComputePvalsCovariance( dim, n,  cov, cov_storage, (float*)C,
                             (float*)pval_c );
    sComputePvalsCovariance( dim, n, rcov, rcov_storage, (float*)C,
                             (float*)pval_r );

    for(i = 0; i < dim; i++)
    {
        for(j = 0; j <= i; j++)
        {
            if (pval_r[i][j] < P_THRESHOLD) errnums++;
        }
    }

    printf("\n\nP-values of the computed classic covariance matrix\n");

    for(i = 0; i < dim; i++)
    {
        for(j = 0; j <= i; j++)
        {
            printf("%9lf ", pval_c[i][j]);
        }
        printf("\n");
    }

    printf("\n\nP-values of the computed robust covariance matrix\n");

    for(i = 0; i < dim; i++)
    {
        for(j = 0; j <= i; j++)
        {
            printf("%9lf ", pval_r[i][j]);
        }
        printf("\n");
    }

    /***** Printing summary of the test *****/
    if (errnums == 0)
    {
        printf("\n\nRobust covariance estimate is agreed with theory\n");
    }
    else
    {
        printf("\n\nError: Robust covariance estimate is ");
        printf("disagreed with theory\n");
        return 1;
    }

    /***** Delete Summary Statistics task *****/
    errcode = vslSSDeleteTask( &task );
    CheckVslError(errcode);

    MKL_Free_Buffers();

    return 0;
}
